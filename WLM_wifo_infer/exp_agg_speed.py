# -*- coding: utf-8 -*-
"""Parameter sweep for the ATA (token aggregation) inference speed.

Isolates the effect of merging on ONE model (the Agg checkpoint):
tau = 0 reproduces the Noagg / enhance_batch pipeline speed, larger tau
+ lower thresholds merge more tokens. Reports for every config:

    avg inference time (ms/sample, batch_num=32)
    token ratio (kept tokens / 192)  <- actual compression achieved
    Channel NMSE (accuracy cost)

Run:  python exp_agg_speed.py [quick]
"""
import importlib.util
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

spec = importlib.util.spec_from_file_location(
    "inf", "inference_GPUfixedSNR_noise_Oct_WiFo2_enhance_batch.py")
inf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inf)

from train_chanest_consis_auto import load_chanest_model_auto

QUICK = "quick" in sys.argv

DATA = "./multi-task-LLM/data_CE/Train/Chest_USRP_Octv1.mat"
CKPT = "./multi-task-LLM/output/enhance/Agg/WLMtra.bin"
BATCH = 32
N_SUB = 256     # only load this many samples (memory limit on this box)


def load_subset(mat_path, n):
    """Same as inf.load_chanest_data but reads only the first n samples
    directly from HDF5 (partial read, low memory)."""
    import h5py
    with h5py.File(mat_path, "r") as f:
        g = f["StateOut4Est"]

        def rc(ds):
            d = ds[..., :n]
            if d.dtype.names:
                return (d["real"] + 1j * d["imag"]).T
            return np.array(d).T

        return (rc(g["rxDmRsRe"]), rc(g["dmrsTx"]), rc(g["dmrsH"]),
                np.array(g["nvar_ideal"])[..., :n].T,
                np.array(g["nvar_est"])[..., :n].T)


print("loading data ...")
DMRS_rcv, DMRS_pilot, DMRS_ch_true, nv_ideal, nv_est = load_subset(DATA, N_SUB)
n_all = DMRS_rcv.shape[0]
print("samples:", n_all)

print("loading model ...")
model = load_chanest_model_auto(CKPT, "cpu")
model.eval()

N_TIMED = 1 if QUICK else 3     # timed batches per config
OFFSET = 32                     # samples [0:32] are the warmup batch


def bench(tau=0, theta_c=None, theta_k=None, batch_num=BATCH, n_batches=N_TIMED,
          offset=OFFSET, static=False, final_compact=False,
          count_syncs=False):
    model.tau_eval = tau
    model.theta_c = theta_c if theta_c is not None else model.theta_c
    model.theta_k = theta_k if theta_k is not None else model.theta_k
    model.ata_static = static
    model.ata_final_compact = final_compact
    ratios, t_sum, nmse_sum, cnt = [], 0.0, 0.0, 0
    n_sync = 0
    for b in range(n_batches):
        i = offset + b * batch_num
        e = min(i + batch_num, n_all)
        pilot, recv = inf.preprocess_chanest(
            DMRS_rcv[i:e], DMRS_pilot[i:e])
        if count_syncs:
            n_sync += count_item_calls(
                lambda: inf.infer_chanest(model, pilot, recv, "cpu"))
            continue
        t0 = time.perf_counter()
        out, dmrs, nvar = inf.infer_chanest(model, pilot, recv, "cpu")
        t_sum += time.perf_counter() - t0
        ratios.append(model.last_token_ratio)
        H, nv = inf.postprocess_chanest(dmrs, nvar)
        nmse_sum += inf.NMSE(np.reshape(DMRS_ch_true[i:e], (-1, 1)),
                             np.reshape(H, (-1, 1)))
        cnt += e - i
    if count_syncs:
        return n_sync
    r = np.array(ratios)
    return t_sum / cnt * 1000.0, float(r.mean()), nmse_sum / n_batches, r


def count_item_calls(fn):
    """Number of Python-visible host syncs inside fn: .item() calls,
    bool()/int() tensor conversions AND boolean-mask indexing (each of
    them forces a GPU->CPU sync on a CUDA device)."""
    real_item = torch.Tensor.item
    real_bool = torch.Tensor.__bool__
    real_getitem = torch.Tensor.__getitem__
    n = [0]

    def count_call(key):
        def _c(*a, **k):
            n[0] += 1
            return key
        return _c

    def item_c(self):
        n[0] += 1
        return real_item(self)

    def bool_c(self):
        n[0] += 1
        return real_bool(self)

    def getitem_c(self, key):
        if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
            n[0] += 1                    # boolean indexing -> nonzero sync
        return real_getitem(self, key)

    torch.Tensor.item = item_c
    torch.Tensor.__bool__ = bool_c
    torch.Tensor.__getitem__ = getitem_c
    try:
        fn()
    finally:
        torch.Tensor.item = real_item
        torch.Tensor.__bool__ = real_bool
        torch.Tensor.__getitem__ = real_getitem
    return n[0]


# ---- warmup (also used to stabilise allocator / lazy init) ----------
print("warmup ...")
bench(0, None, None, n_batches=1, offset=0)

# ---- sync count per batch: dynamic vs single-sync mode ---------------
print("\nsync count (Tensor.item calls = forced GPU->CPU syncs) per batch:")
for name, kw in [
        ("no-merge (tau=0)              ",
         dict(tau=0, static=False, final_compact=False)),
        ("dynamic compaction (old)      ",
         dict(tau=2, theta_c=0.7, theta_k=0.7, static=False)),
        ("single-sync compaction (new)  ",
         dict(tau=2, theta_c=0.7, theta_k=0.7, static=True,
              final_compact=True))]:
    n_sync = bench(count_syncs=True, n_batches=1, **kw)
    print(f"  {name}: {n_sync}")

# ---- timing / accuracy: dynamic vs single-sync ------------------------
print(f"\n{'config':<34}{'ms/sample':>10}{'ratio':>8}"
      f"{'kept/192':>10}{'NMSE':>12}")
print("-" * 74)
rows = [
    ("no-merge baseline (tau=0)", dict(tau=0, static=False)),
    ("tau=2 dynamic (old path)",
     dict(tau=2, theta_c=0.7, theta_k=0.7, static=False)),
    ("tau=2 single-sync (new path)",
     dict(tau=2, theta_c=0.7, theta_k=0.7, static=True,
          final_compact=True)),
    ("tau=2 0.5/0.5 single-sync",
     dict(tau=2, theta_c=0.5, theta_k=0.5, static=True,
          final_compact=True)),
    ("tau=3 0.5/0.5 single-sync",
     dict(tau=3, theta_c=0.5, theta_k=0.5, static=True,
          final_compact=True)),
]
base_ms = None
for name, kw in rows:
    ms, ratio, nmse, _ = bench(**kw)
    if base_ms is None:
        base_ms = ms
    print(f"{name:<34}{ms:>10.2f}{ratio:>8.3f}"
          f"{ratio * 192:>10.1f}{nmse:>12.4g}"
          f"   speedup {base_ms / ms:>5.1f}x")

# ---- batch-padding diagnosis: same config at B=1 vs B=32 -------------
if not QUICK:
    print("\nbatch-padding diagnosis (tau=2, tc=0.7, tk=0.7):")
    ms1, r_mean, nmse1, _ = bench(2, 0.7, 0.7, batch_num=1, n_batches=32,
                                  offset=OFFSET)
    ms32, r32, nmse32, r_arr = bench(2, 0.7, 0.7, batch_num=32)
    print(f"  B=1  : ratio mean {r_mean:.3f}  ms/sample {ms1:.2f}")
    print(f"  B=32 : ratio mean {r32:.3f}  ms/sample {ms32:.2f}")
    print("  -> B=32 batch pads every batch to the WORST sample; the")
    print("     effective compression is the batch-max, not the mean.")
