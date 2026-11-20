"""Fine-tune WiFo-2 (Base-Dense) with the ConsisFormer+ChannelAttention+learned
ATA criterion pipeline of train_consis_ca_auto.py, on
Chest_PoC16_USRP1019.mat, for the infer_chanest() interface of
inference_GPUfixedSNR_noise_Oct_WiFo2.py.

What this file inherits from each parent
---------------------------------------
Data / inference interface (from train_chanest.py):
    * reads data/Chest_PoC16_USRP1019.mat (StateOut4Est: rxDmRsRe (B,2,2,384),
      dmrsTx (B,1,2,384), dmrsH (B,2,2,768), nvar_ideal);
    * the row layout of preprocess_chanest()/postprocess_chanest() of the
      Oct inference script: per-sample 8 rows (half, symbol, rx), each with
      192 comb-2 pilots -> model returns (output_hat, DMRS_est, Noise_power_est)
      with DMRS_est in H_true/H_MAX units (H_MAX = 0.0011) and the analytic
      per-row noise variance;
    * the WiFo-2 backbone (basepro1) initialised from ../Base-dense.

Improvements over train_chanest.py (from train_consis_ca_auto.py / L3):
    1. ATA adaptive token aggregation: tau rounds merging adjacent raw
       tokens whose normalized complex similarity exceeds a threshold
       (geometric-mean merge, fractional positions, validity mask);
    2. LEARNED, input-dependent merge criterion (ATACriterion):
       rho(df) kernel correlation gating the merges (hybrid / kernel /
       content modes), trained by the auxiliary alignment loss
       MSE(rho, s.detach());
    3. Channel Attention installed in every attention layer
       (CAMaskedAttentionWrapper: key-padding mask + MPC kernel bias,
       per-example rate powers g(x), tanh-capped blend lam, u_w query
       bias; lam = 0 at init reproduces the plain attention);
    4. KernelWeightedFSI: MMSE-style physics-informed interpolation that
       recovers the full-length feature sequence from the sparse merged
       tokens (kernel-biased attention over fractional positions);
    5. fractional position embeddings (pos_embed_frac) for the merged
       token positions;
    6. AAE tau sampling: tau ~ U[0, consis_tau_max] per training step and a
       fixed evaluation tau (the runtime compute-performance knob);
    7. loss = supervised channel MSE (grid units) + auto_beta * alignment
       loss; token ratio / criterion / |lam| monitored per epoch;
    8. checkpoint stores the full wrapper state (wifo + fsi + criterion);
       --resume supported.

Usage
-----
Train (the best checkpoint is auto-saved as <out_dir>/model_best.bin):
    python train_chanest_consis_auto.py
    python train_chanest_consis_auto.py --consis_eval_tau 3   # cheap mode
    python train_chanest_consis_auto.py --save_name model_best.pkl  # other ext
    python train_chanest_consis_auto.py --eval_only --resume <ckpt>

The checkpoint is the full wrapper state_dict (wifo + fsi + criterion);
torch.save/torch.load are extension-agnostic, so --save_name accepts any
file name/extension (.pkl/.pt/.bin/...). --resume accepts the same files.

Use the trained checkpoint with inference_GPUfixedSNR_noise_Oct_WiFo2.py
(replace its load_model with this loader; preprocess_chanest, infer_chanest
and postprocess_chanest stay untouched - the calling convention is
identical to train_chanest.py):

    from train_chanest_consis_auto import load_chanest_model_auto
    model = load_chanest_model_auto(
        "./WiFo-2-main/experiments/Chest_PoC16_chanest_consis_auto/model_best.bin",
        device)
    ...
    H_est_complex, noise_var_est = inference_chanest(
        model, DMRS_rcv[i:i+1], DMRS_pilot[i:i+1], device=device)
"""
# coding=utf-8

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_chanest import (
    ChanestDataset,
    ChanestModel,
    H_MAX,
    build_rows,
    create_argparser as create_chanest_argparser,
    read_stateout,
    resolve_device,
    safe_load_state_dict,
    setup_seed,
    str2bool,
)
from train_consis_ca import (
    ChannelAttentionFrac,
    install_ca_masked_attention,
    pos_embed_frac,
    pos_embed_frac_gpu,
)
from train_consis_ca_mmse import KernelWeightedFSI
from train_consis_ca_auto import ATACriterion, run_ata_learned


# ---------------------------------------------------------------------------
# Model wrapper: WiFo-2 + ConsisFormer/CA/learned-ATA behind the
# infer_chanest() calling convention
# ---------------------------------------------------------------------------
class ChanestModelAuto(ChanestModel):
    """ChanestModel whose grid prediction goes through the level-3 pipeline:

        patchify -> ATA (learned criterion) -> Embedding + fractional pos
        -> CA-wrapped encoder blocks -> decoder path -> KernelWeightedFSI
        -> decoder_pred -> unpatchify

    The forward signature is exactly what infer_chanest() calls:
        forward(pilot_sequence, received_DMRS, instruction=None,
                ta_perform='chanest') -> (output_hat, DMRS_est, Noise_power_est)
    (an optional ``tau`` keyword overrides the stored evaluation tau; the
    inference script never passes it, so self.tau_eval is used).

    The supervised-loss extras of the training loop are exposed through
    ``self.last_crit_loss`` (alignment loss tensor) and
    ``self.last_token_ratio`` (kept-token fraction), both refreshed on every
    forward call.
    """

    def __init__(self, args, n_half=2, n_symbol=2, n_rx=2, h_max=H_MAX):
        super().__init__(args, n_half, n_symbol, n_rx, h_max)
        len_patch = self.wifo.t_patch_size * self.wifo.patch_size ** 2
        self.fsi = KernelWeightedFSI(
            dim=self.wifo.embed_dim, num_heads=args.consis_fsi_heads,
            n_rate=args.fsi_ca_n_rate, rate_max=args.fsi_ca_rate_max,
            lambda_max=args.fsi_ca_lambda_max,
            ue_gating=args.fsi_ca_ue_gating)
        self.criterion = ATACriterion(
            feat_dim=2 * len_patch, n_rate=args.auto_n_rate,
            rate_max=args.auto_rate_max)
        self.holder = {}
        self.tau_eval = int(args.consis_eval_tau)
        self.theta_c = float(args.consis_theta)
        self.theta_k = float(args.auto_theta_k)
        self.auto_mode = args.auto_criterion
        self.last_crit_loss = None
        self._last_token_ratio_t = None
        # Inference latency switches (improvements 3+4):
        #   ata_static    - sync-free, fixed-length ATA compaction (set
        #                    True by the inference scripts; does not
        #                    change the outputs at all: same tokens,
        #                    front-packed, trailing invalid padding)
        #   pos_embed_gpu - fractional pos embedding computed on the
        #                    device (no .cpu()/numpy round trip; only
        #                    ulp-level numeric differences vs numpy)
        # gpu_pos_embed of the FSI is enabled for the same reason.
        self.ata_static = False
        # large-batch inference mode: async merge rounds + ONE final
        # truncation to the batch-max kept length (single sync/batch).
        # Set together with ata_static=True by the *_agg_batch script.
        self.ata_final_compact = False
        self.pos_embed_gpu = True
        self._ca_modules = None
        self.fsi.gpu_pos_embed = True

    def _lam_init(self, lam_init):
        """Fill the tanh-capped lambda of every CA module (train_ca.py
        semantics: lam = lambda_max * tanh(raw) -> invert with atanh)."""
        for module in self.modules():
            ca = getattr(module, "channel_attention", None)
            if ca is None:
                continue
            if ca.lambda_max and ca.lambda_max > 0:
                ratio = min(abs(lam_init) / ca.lambda_max, 0.999)
                raw = ca.lambda_max * math.atanh(ratio)
                raw = math.copysign(raw, lam_init)
            else:
                raw = lam_init
            ca.lam.data.fill_(raw)

    @property
    def last_token_ratio(self):
        """Kept-token fraction of the last forward. Backed by a device
        tensor, so no .item() sync happens inside forward(); the
        conversion occurs only when a caller (training/eval logging)
        actually reads this property."""
        t = self._last_token_ratio_t
        return 1.0 if t is None else float(t.item())

    def forward(self, pilot_sequence, received_DMRS, instruction=None,
                ta_perform="chanest", tau=None):
        if ta_perform is None or not str(ta_perform).startswith("chanest"):
            raise NotImplementedError(
                f"ChanestModelAuto only supports ta_perform='chanest', got "
                f"{ta_perform!r}.")

        pilot = torch.as_tensor(pilot_sequence)
        received = torch.as_tensor(received_DMRS)
        if pilot.dim() != 4 or pilot.shape[-1] != 2:
            raise ValueError(
                f"pilot_sequence must be (R, 1, 192, 2), got {tuple(pilot.shape)}")
        R = pilot.shape[0]
        if R % self.rows_per_sample:
            raise ValueError(
                f"Number of rows ({R}) must be a multiple of "
                f"{self.rows_per_sample} (half x symbol x rx).")
        B = R // self.rows_per_sample

        p_c, y_c, inp = self._rows_to_grid(pilot, received)

        # WiFo-2 grid [B*n_rx, 2, T, H, W]: the receive-antenna axis is
        # folded into the BATCH (B*n_rx examples, = 2 with n_rx=2), and the
        # H axis holds the n_symbol rows duplicated once (H = 2*n_symbol
        # = 4, keeping patch_size=4 valid); W = 768 subcarriers.  Each
        # example now carries ONE antenna instead of the (symbol x rx)
        # row stack.
        inp_ab = (inp.permute(0, 2, 1, 3)
                    .reshape(B * self.n_rx, self.n_symbol, inp.shape[-1]))
        inp_h = inp_ab.repeat_interleave(2, dim=1)   # duplicate symbol axis
        grid = (torch.stack([inp_h.real, inp_h.imag], dim=1)
                * self.internal_scale)                    # (B*n_rx, 2, H, W)
        T = self.wifo.args.t_patch_size
        grid = (grid.unsqueeze(2)
                    .expand(-1, -1, T, -1, -1)
                    .contiguous())              # (B*n_rx, 2, T, 4, 768)

        wifo = self.wifo
        n_ex = grid.shape[0]
        raw = wifo.patchify(grid).float()        # [N, L, 2, F] (sets patch_info)
        length = raw.shape[1]

        if tau is None:
            tau = self.tau_eval
        tok, pos, valid, crit_loss = run_ata_learned(
            raw, self.criterion, tau, self.theta_c, self.theta_k,
            self.auto_mode, static_shape=self.ata_static,
            final_compaction=self.ata_final_compact)
        self.last_crit_loss = crit_loss
        # kept as a device tensor; the .item() sync happens only if a
        # caller actually reads the last_token_ratio property
        self._last_token_ratio_t = (valid.float().sum()
                                    / float(n_ex * length))
        l_sp = tok.shape[1]

        # Static kernel-table bound for every CA module (ATA positions
        # stay inside [0, length-1] merged or not): removes the
        # per-call .item() sync + rate-table rebuild of kernel_from_diff.
        if self._ca_modules is None:
            self._ca_modules = [m for m in self.modules()
                                if isinstance(m, ChannelAttentionFrac)]
            for m in self._ca_modules:
                m.diff_max = float(length - 1)

        pe_fn = (pos_embed_frac_gpu if self.pos_embed_gpu
                 else pos_embed_frac)
        x = wifo.Embedding_encoder_CE(tok.reshape(n_ex, l_sp, -1))
        x = x + pe_fn(wifo, pos)
        self.holder["key_padding"] = valid
        self.holder["positions"] = pos
        for blk in wifo.blocks:
            x = blk(x)
        x = wifo.norm(x)
        x = wifo.decoder_embed(x)
        x = x + pe_fn(wifo, pos)
        for blk in wifo.decoder_blocks:
            x = blk(x)
        x = wifo.decoder_norm(x)                 # sparse features V
        self.holder["key_padding"] = None
        self.holder["positions"] = None

        t_emb = wifo.Embedding_encoder_CE(raw.reshape(n_ex, length, -1))
        pos_orig = torch.arange(length, device=x.device, dtype=torch.float32)
        pos_orig = pos_orig.unsqueeze(0).expand(n_ex, -1)
        x_full = self.fsi(x, valid, t_emb, pos_orig, pos)

        pred = wifo.decoder_pred(x_full).float()          # [N, L, 2*F]
        len_patch = wifo.t_patch_size * wifo.patch_size ** 2
        pred_full = wifo.unpatchify(
            pred[..., :len_patch] + 1j * pred[..., len_patch:])  # (B,2,T,H,W)
        pred_full = pred_full.mean(dim=2)                          # avg tiled T

        # (B*n_rx, 2, H, W) real -> (B*n_rx, sym, half*384) complex:
        # first average the duplicated symbol rows back together, then
        # fold the antenna batch axis back into the (half, symbol, rx)
        # row layout expected by postprocess_chanest().
        pred_c = torch.complex(pred_full[:, 0], pred_full[:, 1])  # (B*n_rx, H, W)
        pred_c = (pred_c.reshape(n_ex, self.n_symbol, 2, pred_c.shape[-1])
                        .mean(dim=2))                    # avg duplicated sym
        pred_c = pred_c.reshape(B, self.n_rx, self.n_symbol,
                                self.n_half, -1)
        rows = pred_c.permute(0, 3, 2, 1, 4).reshape(R, -1)  # (b,half,sym,rx)

        # back to the units expected by postprocess_chanest (H_true / H_MAX)
        dmrs = rows / (self.internal_scale * self.h_max)
        dmrs_r = torch.stack([dmrs.real, dmrs.imag], dim=-1).reshape(
            R, 1, dmrs.shape[-1], 2)                       # (R, 1, 384, 2)

        # analytic per-row noise variance from the DMRS residual
        with torch.no_grad():
            h_hat = (rows / self.internal_scale).detach()  # physical units
            pil_rows = p_c.reshape(R, -1)                   # (R, 192)
            rx_rows = y_c.reshape(R, -1)                    # (R, 192)
            res = rx_rows - 0.5 * h_hat[..., 0::2] * torch.conj(pil_rows)
            nvar = res.abs().pow(2).mean(dim=-1, keepdim=True)  # (R, 1)

        output_hat = torch.cat([dmrs_r.reshape(R, -1), nvar], dim=1)
        return output_hat, dmrs_r, nvar


def load_chanest_model_auto(checkpoint_path, device="cpu"):
    """Rebuild a trained ChanestModelAuto (for the inference side).

    Reads the config.json saved next to the checkpoint, installs the CA
    attention wrappers (required BEFORE load_state_dict: wrapping changes
    the state-dict key names) and restores the full wrapper state.
    """
    checkpoint_path = Path(checkpoint_path)
    cfg_path = checkpoint_path.parent / "config.json"
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    args = argparse.Namespace(
        size=cfg.get("size", "basepro1"),
        MoE=cfg.get("MoE", False),
        pos_emb=cfg.get("pos_emb", "SinCos_3D"),
        no_qkv_bias=cfg.get("no_qkv_bias", 0),
        t_patch_size=cfg.get("t_patch_size", 4),
        patch_size=cfg.get("patch_size", 4),
        device_id=str(cfg.get("device_id", "cpu")),
        # ConsisFormer / CA / FSI / learned-criterion defaults
        consis_theta=cfg.get("consis_theta", 0.985),
        consis_eval_tau=cfg.get("consis_eval_tau", 0),
        consis_fsi_heads=cfg.get("consis_fsi_heads", 8),
        fsi_ca_n_rate=cfg.get("fsi_ca_n_rate", 16),
        fsi_ca_rate_max=cfg.get("fsi_ca_rate_max", 0.5),
        fsi_ca_lambda_max=cfg.get("fsi_ca_lambda_max", 1.0),
        fsi_ca_ue_gating=cfg.get("fsi_ca_ue_gating", True),
        auto_n_rate=cfg.get("auto_n_rate", 16),
        auto_rate_max=cfg.get("auto_rate_max", 0.5),
        auto_theta_k=cfg.get("auto_theta_k", 0.97),
        auto_criterion=cfg.get("auto_criterion", "hybrid"),
    )
    model = ChanestModelAuto(
        args, n_half=cfg.get("n_half", 2), n_symbol=cfg.get("n_symbol", 2),
        n_rx=cfg.get("n_rx", 2), h_max=cfg.get("h_max", H_MAX))
    install_ca_masked_attention(
        model.wifo, model.holder, n_rate=cfg.get("ca_n_rate", 16),
        rate_max=cfg.get("ca_rate_max", 0.5),
        lambda_max=cfg.get("ca_lambda_max", 1.0),
        ue_gating=cfg.get("ca_ue_gating", True),
        layers=cfg.get("ca_layers", "all"))
    try:
        state = torch.load(checkpoint_path, map_location=device,
                           weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def forward_chanest_auto(model, pilot, received, target, snr_db, tau):
    """One supervised step: call the model exactly like infer_chanest() does,
    with the AAE tau for this step, and build the combined loss
    (grid-unit channel MSE + auto_beta is applied by the caller)."""
    if snr_db is not None:
        power = (received ** 2).mean()
        received = received + torch.randn_like(received) * torch.sqrt(
            power * 10 ** (-snr_db / 10.0))
    output_hat, dmrs_est, nvar_est = model(
        pilot_sequence=pilot.reshape(-1, 1, pilot.shape[-2], 2),
        received_DMRS=received.reshape(-1, 1, received.shape[-2], 2),
        instruction=None,
        ta_perform="chanest",
        tau=tau,
    )
    tgt = target.reshape(-1, 1, target.shape[-2], 2)
    # rescale the channel MSE into the normalised grid units used by the
    # level-3 scripts, so auto_beta has the same meaning there
    unit = float(model.internal_scale * model.h_max)
    loss_chan = F.mse_loss(dmrs_est, tgt) * (unit ** 2)
    rows = pilot.shape[1]
    d = rows * tgt.shape[-2] * 2
    err = ((dmrs_est - tgt) ** 2).reshape(-1, d).sum(1)
    ref = (tgt ** 2).reshape(-1, d).sum(1)
    nmse = (err / ref.clamp_min(1e-12)).mean().item()
    return loss_chan, nmse, dmrs_est, nvar_est


@torch.inference_mode()
def evaluate_auto(model, loader, device, max_batches=0):
    """Mean per-sample channel NMSE (+ noise NMSE + token ratio)."""
    model.eval()
    total_nmse, total_noise, total_ratio, count = 0.0, 0.0, 0.0, 0
    for i, (pilot, received, target, nvar) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        pilot = pilot.to(device, non_blocking=True)
        received = received.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        nvar = nvar.to(device, non_blocking=True)
        _, dmrs_est, nvar_est = model(
            pilot_sequence=pilot.reshape(-1, 1, pilot.shape[-2], 2),
            received_DMRS=received.reshape(-1, 1, received.shape[-2], 2),
            instruction=None, ta_perform="chanest")
        tgt = target.reshape(-1, 1, target.shape[-2], 2)
        rows = pilot.shape[1]
        d = rows * tgt.shape[-2] * 2
        err = ((dmrs_est - tgt) ** 2).reshape(-1, d).sum(1)
        ref = (tgt ** 2).reshape(-1, d).sum(1)
        total_nmse += (err / ref.clamp_min(1e-12)).mean().item()
        nvar_pred = nvar_est.reshape(-1, rows).mean(dim=1)
        total_noise += (((nvar_pred - nvar) ** 2)
                        / (nvar ** 2).clamp_min(1e-20)).mean().item()
        total_ratio += model.last_token_ratio
        count += 1
    model.train()
    if count == 0:
        raise ValueError("Evaluation loader is empty.")
    nmse = total_nmse / count
    return (nmse, 10.0 * math.log10(max(nmse, 1e-12)),
            total_noise / count, total_ratio / count)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def create_auto_argparser():
    # Extend (not parents-wrap) the chanest parser: a parents=[...] chain
    # would duplicate -h/--help.
    parser = create_chanest_argparser(add_help=False)
    parser.description = ("Level-3 ConsisFormer + Channel Attention + learned "
                          "ATA criterion on the chanest inference interface.")
    parser.set_defaults(out_dir="./experiments/Chest_PoC16_chanest_consis_auto")
    # (--save_name is inherited from the train_chanest base parser)
    # --- ConsisFormer (train_consis_ca.py) ---
    parser.add_argument("--consis_theta", type=float, default=0.8,
                        help="ATA merge threshold on the normalized complex "
                             "inner product (content criterion).")
    parser.add_argument("--consis_tau_max", type=int, default=3,
                        help="Maximum ATA rounds; tau is sampled from "
                             "U[0, tau_max] at every training step (AAE).")
    parser.add_argument("--consis_eval_tau", type=int, default=2,
                        help="Fixed ATA rounds at evaluation (the runtime "
                             "compute-performance knob).")
    parser.add_argument("--consis_fsi_heads", type=int, default=8,
                        help="Attention heads in the FSI interpolation.")
    # --- Channel Attention (train_consis_ca.py) ---
    parser.add_argument("--ca_n_rate", type=int, default=16,
                        help="Number of candidate phase rates in the MPC "
                             "kernel (paper's d_t grid).")
    parser.add_argument("--ca_rate_max", type=float, default=0.5,
                        help="Maximum candidate phase rate (radians per "
                             "token step) along the frequency axis.")
    parser.add_argument("--ca_lambda_init", type=float, default=0.0,
                        help="Initial per-head blend of the MPC kernel; 0 "
                             "keeps the exact pretrained behaviour at start.")
    parser.add_argument("--ca_lambda_max", type=float, default=1.0,
                        help="Cap on the effective blend (0 disables).")
    parser.add_argument("--ca_ue_gating", type=str2bool, default=True,
                        help="Enable the per-example D_UE gating g(x).")
    parser.add_argument("--ca_layers", default="all",
                        choices=("all", "enc", "dec"),
                        help="Which attention layers get the CA module.")
    # --- Kernel-weighted FSI (train_consis_ca_mmse.py) ---
    parser.add_argument("--fsi_ca_n_rate", type=int, default=16,
                        help="Candidate rates in the FSI interpolation "
                             "kernel.")
    parser.add_argument("--fsi_ca_rate_max", type=float, default=0.5,
                        help="Max rate of the FSI interpolation kernel.")
    parser.add_argument("--fsi_ca_lambda_init", type=float, default=0.0,
                        help="Initial blend of the FSI kernel bias; 0 "
                             "reduces the FSI exactly to the level-1 FSI.")
    parser.add_argument("--fsi_ca_lambda_max", type=float, default=1.0,
                        help="Cap on the FSI kernel blend.")
    parser.add_argument("--fsi_ca_ue_gating", type=str2bool, default=True,
                        help="Per-example rate powers g(x) in the FSI "
                             "kernel.")
    # --- Learned criterion (train_consis_ca_auto.py) ---
    parser.add_argument("--auto_criterion", default="hybrid",
                        choices=("hybrid", "kernel", "content"),
                        help="Merge rule: hybrid = content similarity AND "
                             "learned kernel gate; kernel = learned "
                             "criterion only; content = level-1/2 fallback.")
    parser.add_argument("--auto_theta_k", type=float, default=0.8,
                        help="Threshold on the learned kernel correlation "
                             "rho. In hybrid mode rho acts as a per-example "
                             "gate (smooth channel -> merging allowed).")
    parser.add_argument("--auto_beta", type=float, default=1.0,
                        help="Weight of the auxiliary alignment loss "
                             "MSE(rho, s) that trains the criterion.")
    parser.add_argument("--auto_n_rate", type=int, default=16,
                        help="Candidate rates in the criterion kernel.")
    parser.add_argument("--auto_rate_max", type=float, default=0.5,
                        help="Max rate of the criterion kernel.")
    # --- Resume ---
    parser.add_argument("--resume", default="",
                        help="Path to a saved checkpoint to continue from "
                             "or evaluate (with --eval_only).")
    return parser


def _fill_lam(ca, lam_init):
    """Fill one CA module's tanh-capped lambda (lam = lambda_max*tanh(raw))."""
    if ca.lambda_max and ca.lambda_max > 0:
        ratio = min(abs(lam_init) / ca.lambda_max, 0.999)
        raw = ca.lambda_max * math.atanh(ratio)
        raw = math.copysign(raw, lam_init)
    else:
        raw = lam_init
    ca.lam.data.fill_(raw)


def main():
    args = create_auto_argparser().parse_args()
    setup_seed(args.seed)
    device = resolve_device(args.device_id)
    args.device_id = str(args.device_id)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_cuda = device.type == "cuda"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data (identical pipeline to train_chanest.py)
    # ------------------------------------------------------------------
    print(f"Loading data from {args.data_path} ...")
    fields = read_stateout(args.data_path)
    pilot, received, target, nvar, scale, base_nmse = build_rows(
        fields, n_half=args.n_half, n_symbol=args.n_symbol, n_rx=args.n_rx)
    del fields
    G = pilot.shape[0]
    n_train = int(args.train_ratio * G)
    n_val = int(args.val_ratio * G)
    n_test = G - n_train - n_val
    print(f"Samples: {G} total -> train {n_train}, val {n_val}, test {n_test}")
    print(f"[baseline  ] LS+linear-interp NMSE = {base_nmse:.6f} "
          f"({10.0 * math.log10(max(base_nmse, 1e-12)):.4f} dB)")

    if args.train_subset:
        n_train = min(args.train_subset, n_train)

    train_set = ChanestDataset(pilot[:n_train], received[:n_train],
                               target[:n_train], nvar[:n_train])
    val_set = ChanestDataset(pilot[n_train:n_train + n_val],
                             received[n_train:n_train + n_val],
                             target[n_train:n_train + n_val],
                             nvar[n_train:n_train + n_val])
    test_set = ChanestDataset(pilot[n_train + n_val:],
                              received[n_train + n_val:],
                              target[n_train + n_val:],
                              nvar[n_train + n_val:])

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=use_cuda)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=use_cuda)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=use_cuda)

    # ------------------------------------------------------------------
    # Model: backbone + CA/masked attention + kernel FSI + ATACriterion
    # ------------------------------------------------------------------
    model = ChanestModelAuto(args, n_half=args.n_half, n_symbol=args.n_symbol,
                             n_rx=args.n_rx, h_max=args.h_max).to(device)
    model.internal_scale.fill_(scale)

    # Load the pretrained backbone BEFORE wrapping: install_ca_masked_attention
    # changes the state-dict key names (Block.attn -> Block.attn.attn).
    checkpoint = Path(args.file_load_path)
    if checkpoint.suffix not in (".pkl", ".pt", ".pth", ".bin"):
        checkpoint = Path(f"{checkpoint}.pkl")
    if checkpoint.is_file():
        state_dict = safe_load_state_dict(checkpoint, device)
        incompatible = model.wifo.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"Missing checkpoint keys: {incompatible.missing_keys}")
            print(f"Unexpected checkpoint keys: "
                  f"{incompatible.unexpected_keys}")
        print(f"Loaded backbone: {checkpoint}")
    else:
        print(f"Checkpoint not found ({checkpoint}); training from scratch.")

    adapted = install_ca_masked_attention(
        model.wifo, model.holder, n_rate=args.ca_n_rate,
        rate_max=args.ca_rate_max, lambda_max=args.ca_lambda_max,
        ue_gating=args.ca_ue_gating, layers=args.ca_layers)
    # install_ca_masked_attention attaches freshly-built CA modules that
    # were created on the CPU AFTER the .to(device) above; move them to
    # the target device now, otherwise the first forward crashes with a
    # cpu/cuda mismatch inside ChannelAttentionFrac.rate_powers (W_g on
    # cpu vs activations on cuda).
    model.to(device)

    if args.ca_lambda_init != 0.0:
        for module in model.modules():
            ca = getattr(module, "channel_attention", None)
            if ca is not None and ca is not model.fsi.ca:
                _fill_lam(ca, args.ca_lambda_init)
    if args.fsi_ca_lambda_init != 0.0:
        _fill_lam(model.fsi.ca, args.fsi_ca_lambda_init)

    n_crit = sum(p.numel() for p in model.criterion.parameters())
    print(f"Level-3 combination: criterion={args.auto_criterion}, "
          f"theta_c={args.consis_theta}, theta_k={args.auto_theta_k}, "
          f"beta={args.auto_beta}, tau_max={args.consis_tau_max} "
          f"(+{n_crit:,} criterion params, CA layers: {adapted})")

    if args.resume and Path(args.resume).is_file():
        state = safe_load_state_dict(args.resume, device)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Resume file does not match the model structure "
                f"(missing {incompatible.missing_keys[:3]}, unexpected "
                f"{incompatible.unexpected_keys[:3]}).")
        print(f"Resumed checkpoint from {args.resume}")

    config = vars(args).copy()
    config["internal_scale"] = scale
    config["baseline_nmse"] = base_nmse
    config["ca_adapted_modules"] = adapted
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Zero-shot performance before fine-tuning
    # ------------------------------------------------------------------
    nmse, val_db, noise_nmse, tok_ratio = evaluate_auto(
        model, val_loader, device, max_batches=args.eval_subset)
    print(f"[zero-shot ] val  NMSE = {nmse:.6f} ({val_db:.4f} dB) "
          f"noise NMSE = {noise_nmse:.6f} tok = {tok_ratio:.2f} "
          f"[eval tau={args.consis_eval_tau}]")

    if args.eval_only:
        nmse, test_db, noise_nmse, tok_ratio = evaluate_auto(
            model, test_loader, device, max_batches=args.eval_subset)
        print(f"[auto L3   ] test NMSE = {nmse:.6f} ({test_db:.4f} dB) "
              f"noise NMSE = {noise_nmse:.6f} tok = {tok_ratio:.2f} "
              f"[eval tau={args.consis_eval_tau}]")
        return

    # ------------------------------------------------------------------
    # Training loop (all modules + criterion; random tau per step)
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max(args.epochs, 1) * steps_per_epoch
    warmup_steps = max(args.warmup_epochs, 0) * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = args.amp and device.type == "cuda"

    best_val = float("inf")
    snr_aug = None if args.snr_db < 0 else args.snr_db
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss, seen, t0 = 0.0, 0, time.time()
        ratio_sum, crit_sum, n_ratio = 0.0, 0.0, 0
        for pilot_b, received_b, target_b, _ in train_loader:
            tau = random.randint(0, args.consis_tau_max)  # AAE sampling
            pilot_b = pilot_b.to(device, non_blocking=True)
            received_b = received_b.to(device, non_blocking=True)
            target_b = target_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                loss_chan, _, _, _ = forward_chanest_auto(
                    model, pilot_b, received_b, target_b, snr_aug, tau)
                loss = loss_chan
                if model.last_crit_loss is not None:
                    loss = loss + args.auto_beta * model.last_crit_loss
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * pilot_b.shape[0]
            seen += pilot_b.shape[0]
            ratio_sum += model.last_token_ratio
            crit_sum += (float(model.last_crit_loss.detach())
                         if model.last_crit_loss is not None else 0.0)
            n_ratio += 1
        train_loss = epoch_loss / max(seen, 1)
        ratio_mean = ratio_sum / max(n_ratio, 1)
        crit_mean = crit_sum / max(n_ratio, 1)

        lam_mean, n_ca = 0.0, 0
        for module in model.modules():
            ca = getattr(module, "channel_attention", None)
            if ca is not None:
                lam_mean += (ca.effective_lambda().detach()
                             .abs().mean().item())
                n_ca += 1
        if n_ca:
            lam_mean /= n_ca

        nmse, val_db, noise_nmse, tok_ratio = evaluate_auto(
            model, val_loader, device, max_batches=args.eval_subset)
        marker = ""
        if nmse < best_val:
            best_val = nmse
            torch.save(model.state_dict(), out_dir / args.save_name)
            marker = "  <- saved"
        print(f"[epoch {epoch:3d}/{args.epochs}] loss={train_loss:.6f} "
              f"val NMSE={nmse:.6f} ({val_db:.4f} dB) "
              f"noise NMSE={noise_nmse:.6f} "
              f"tok={ratio_mean:.2f} crit={crit_mean:.4f} "
              f"|lam|={lam_mean:.3f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} "
              f"({time.time() - t0:.1f}s){marker}")

    # ------------------------------------------------------------------
    # Final test with the best checkpoint
    # ------------------------------------------------------------------
    best_path = out_dir / args.save_name
    if best_path.is_file():
        model.load_state_dict(safe_load_state_dict(best_path, device))
    nmse, test_db, noise_nmse, tok_ratio = evaluate_auto(
        model, test_loader, device, max_batches=args.eval_subset)
    print(f"[best L3   ] test NMSE = {nmse:.6f} ({test_db:.4f} dB) "
          f"noise NMSE = {noise_nmse:.6f} tok = {tok_ratio:.2f} "
          f"[eval tau={args.consis_eval_tau}]")
    print(f"Best checkpoint saved to {best_path}")
    print("Use it with inference_GPUfixedSNR_noise_Oct_WiFo2.py via "
          "train_chanest_consis_auto.load_chanest_model_auto().")


if __name__ == "__main__":
    main()






