"""Batched-inference variant of inference_GPUfixedSNR_noise_Oct_WiFo2.py.

Identical to the original script except that the channel-estimation
and PoC-detection main loops call the model on batch_num samples per
forward pass (batch_num is defined at the top of __main__). All outputs
(per-sample NMSE statistics, CE_result/CE_output/DET_result/DET_output
.mat contents) are unchanged - with batch_num = 1 the behaviour is
identical to the original per-sample scripts.
"""


import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io
#from models.gpt2_model import Gpt2Model
from train_chanest import load_chanest_model, patch_ce_interpolation
#from train_chanest_consis_auto import load_chanest_model_auto
from einops import rearrange
import h5py
from scipy.io import savemat
import time


# ============= Model Parameter =============#
class Params:
    def __init__(self):
        self.model_path = "./gpt-ori"
        self.predict_dimension_in = 48
        self.predict_dimension_out = 48
        self.prev_len = 16
        self.pred_len = 4
        self.muh_dimen = 128
        self.sigma = 10**(-10/10)



# ============= load model =============#
def _patch_pos_embed_cache(wifo):
    """Cache the SinCos_3D position embeddings on the device.

    pos_embed_enc_3d()/pos_embed_dec_3d() otherwise recompute the SAME
    numpy sincos table on the CPU and copy it to the GPU on EVERY forward
    call - a fixed per-call cost independent of the batch size (the chanest
    input size (T,H,W) is constant). CE inference always uses
    ids_keep=arange(L), so one cached tensor is reused for every call."""
    import types
    orig_enc = type(wifo).pos_embed_enc_3d
    orig_dec = type(wifo).pos_embed_dec_3d
    cache = {}

    def cached_enc(self, ids_keep, batch, input_size, scale):
        key = ("enc", input_size, tuple(scale))
        if key not in cache:
            cache[key] = orig_enc(self, ids_keep[:1], 1, input_size, scale)
        return cache[key].expand(batch, -1, -1)

    def cached_dec(self, ids_keep, batch, input_size, scale):
        key = ("dec", input_size, tuple(scale))
        if key not in cache:
            cache[key] = orig_dec(self, ids_keep[:1], 1, input_size, scale)
        return cache[key].expand(batch, -1, -1)

    wifo.pos_embed_enc_3d = types.MethodType(cached_enc, wifo)
    wifo.pos_embed_dec_3d = types.MethodType(cached_dec, wifo)


def load_model(model_path, device='cpu'):
    #params = Params()
    #model = Gpt2Model(params)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    model = load_chanest_model(model_path, device)
    # Fast-path for identity CE strides (1,1,1): skip the per-forward
    # CPU/Delaunay scattered-interpolation round trip (it is an identity
    # map anyway). Same patch the training loop of train_chanest.py
    # installs, but load_chanest_model() does not apply it.
    patch_ce_interpolation(model.wifo)
    _patch_pos_embed_cache(model.wifo)
    model.to(device)
    model.eval()
    return model


def compute_noise_variance_nvar(H_est_complex, DMRS_rcv, DMRS_pilot, nRx=2):

    nRB = 64
    nRE_full = 12
    nRE_dmrs = 6
    Q15 = 32768.0
    Q30 = 1073741824.0

    B = H_est_complex.shape[0]

    nRx_actual = H_est_complex.shape[1]
    nSymbol_actual = H_est_complex.shape[2]
    H_reshaped = H_est_complex.reshape(B, nRx_actual, nSymbol_actual, nRB, nRE_full)
    dmrs_indices = [0, 2, 4, 6, 8, 10]
    H_at_dmrs = H_reshaped[:, :, :, :, dmrs_indices]
    H_at_dmrs = H_at_dmrs.reshape(B, nRx_actual, nSymbol_actual, nRB * nRE_dmrs)

    H_q15 = H_at_dmrs * Q15
    if isinstance(DMRS_rcv, torch.Tensor):
        rx_q15 = DMRS_rcv.cpu().numpy() * Q15
        p_q15 = DMRS_pilot.cpu().numpy() * Q15 / 2
    else:
        rx_q15 = DMRS_rcv * Q15
        p_q15 = DMRS_pilot * Q15 / 2

    # H_q15 shape: (B, nRx, nSymbol, nRBRE)
    # rx_q15 shape: (1, nRx, nSymbol, nRBRE) or (B, nRx, nSymbol, nRBRE)
    # p_q15 shape: (1, 1, nSymbol, nRBRE)

    H_re = H_q15.real  # (B, nRx, nSymbol, nRBRE)
    H_im = H_q15.imag

    rx_re = rx_q15[:, :, :, :].real  # (B, nRx, nSymbol, nRBRE)
    rx_im = rx_q15[:, :, :, :].imag

    p_re = p_q15[:, 0, :, :].real  # (B, nSymbol, nRBRE) - broadcast to all nRx
    p_im = p_q15[:, 0, :, :].imag

    # Broadcast p to all nRx: (B, nRx, nSymbol, nRBRE).
    # NOTE: insert the singleton nRx axis FIRST - broadcasting the 3-D
    # (B, nSymbol, nRBRE) tensor directly only worked for B == 1 (its
    # batch axis was silently remapped onto nRx) and raises a
    # ValueError for any larger batch.
    p_re = np.broadcast_to(p_re[:, None], (B, nRx, nSymbol_actual, nRB * nRE_dmrs))
    p_im = np.broadcast_to(p_im[:, None], (B, nRx, nSymbol_actual, nRB * nRE_dmrs))

    # Hls calculation (vectorized for all nRx)
    Hls_re = np.round(((rx_re * p_re - rx_im * p_im) / Q15)).astype(np.int16)
    Hls_im = np.round(((rx_re * p_im + rx_im * p_re) / Q15)).astype(np.int16)

    # Residual (vectorized)
    res_re = Hls_re.astype(np.int32) - H_re.astype(np.int32)
    res_im = Hls_im.astype(np.int32) - H_im.astype(np.int32)

    # Sum over nSymbol and nRBRE for each (B, nRx)
    noise_amp2 = np.sum(res_re.astype(np.int64)**2 + res_im.astype(np.int64)**2, axis=(2, 3)).astype(float)  # (B, nRx)

    nest_count = 2 * (nSymbol_actual * nRB * nRE_dmrs)
    nvar_per_rx = noise_amp2 / nest_count if nest_count > 0 else 0.0  # (B, nRx)

    # Sum over nRx
    total_nvar = np.sum(nvar_per_rx, axis=1)  # (B,)

    total_nvar /= (nRx * 13)

    nvar_q30 = np.minimum(total_nvar, 4294967295.0)
    output_array_nvar = np.uint32(nvar_q30.astype(np.int64))

    noise_var_est = total_nvar / Q30
    noise_var_est = noise_var_est.reshape(1, B)

    return noise_var_est

def preprocess_chanest(DMRS_rcv, DMRS_pilot):

    nRB = 64
    nRE = 6

    B, nRx, nSymbol, nRBRE = DMRS_rcv.shape
    DMRS_rcv_np = np.zeros((B, nRx, nSymbol, nRBRE, 2))
    DMRS_rcv_np[..., 0] = DMRS_rcv.real
    DMRS_rcv_np[..., 1] = DMRS_rcv.imag
    DMRS_rcv_np = np.reshape(DMRS_rcv_np, [B, nRx, nSymbol, 2, nRB // 2, nRE, 2])
    DMRS_rcv_np = np.transpose(DMRS_rcv_np, [3, 2, 0, 1, 4, 5, 6])
    #DMRS_rcv_np = np.reshape(DMRS_rcv_np, [2*nSymbol*B, nRx, nRB//2*nRE, 2])
    DMRS_rcv_np = np.reshape(DMRS_rcv_np, [2*nSymbol*B*nRx, 1, nRB//2*nRE, 2])

    B, nTx, nSymbol, nRBRE = DMRS_pilot.shape
    DMRS_pilot_np = np.zeros((B, nTx, nSymbol, nRBRE, 2))
    DMRS_pilot_np[..., 0] = DMRS_pilot.real
    DMRS_pilot_np[..., 1] = DMRS_pilot.imag
    DMRS_pilot_np = np.reshape(DMRS_pilot_np, [B, nTx, nSymbol, 2, nRB // 2, nRE, 2])
    DMRS_pilot_np = np.transpose(DMRS_pilot_np, [3, 2, 0, 1, 4, 5, 6])
    DMRS_pilot_np = np.reshape(DMRS_pilot_np, [2 * nSymbol * B, nTx, nRB // 2 * nRE, 2])
    
    ##copy pilot
    DMRS_pilot_np = np.repeat(DMRS_pilot_np,2,axis=1)
    DMRS_pilot_np = np.reshape(DMRS_pilot_np, [2*nSymbol*B*2, 1, nRB//2*nRE, 2])

    pilot = torch.from_numpy(DMRS_pilot_np).float()
    received_DMRS = torch.from_numpy(DMRS_rcv_np).float()

    return pilot, received_DMRS


def infer_chanest(model, pilot, received_DMRS, device='cpu'):

    pilot = pilot.to(device)
    received_DMRS = received_DMRS.to(device)

    with torch.no_grad():
        output_hat, DMRS_est, Noise_power_est = model(
            pilot_sequence=pilot,
            received_DMRS=received_DMRS,
            instruction=None,
            ta_perform='chanest'
        )

    return output_hat, DMRS_est, Noise_power_est


def postprocess_chanest(DMRS_est, Noise_power_est, nRx=2, nSymbol=2):

    nRB = 64
    nRE = 12

    # DMRS_est: (2*nSymbol*B*nRx, 1, 384, 2)  (B, nRx, nSymbol, 768) complex
    DMRS_est_np = DMRS_est.cpu().numpy()
    total_B = DMRS_est_np.shape[0]
    B = total_B // (2 * nSymbol*nRx)

    DMRS_est_np = np.reshape(DMRS_est_np, [2, nSymbol, B, nRx, nRB // 2, nRE, 2])
    DMRS_est_np = np.transpose(DMRS_est_np, [2, 3, 1, 0, 4, 5, 6])
    DMRS_est_np = np.reshape(DMRS_est_np, [B, nRx, nSymbol, nRB * nRE, 2])

    H_est_complex = DMRS_est_np[..., 0] + 1j * DMRS_est_np[..., 1]
    H_max = 0.0011
    H_est_complex = H_est_complex * H_max

    noise_var_est_np = Noise_power_est.cpu().numpy()
    # Noise_power_est rows arrive per-sample contiguous (batch, half, symbol,
    # rx) from the model; average the rows of every sample - identical to
    # the original single-sample reshape [2*nRx, nSymbol, B, 1] + mean(0, 1)
    noise_var_est = np.mean(noise_var_est_np.reshape(B, 2 * nRx * nSymbol), axis=1)  # (B,)
    noise_var_est = noise_var_est.reshape(1, B)  # (1, B)

    return H_est_complex, noise_var_est


def _regroup_rows_per_sample(x, B, n_half=2, n_symbol=2, n_rx=2):
    """Row order (half, symbol, batch, rx) -> (batch, half, symbol, rx).

    preprocess_chanest() emits rows ordered (half, symbol, batch, rx), which
    coincides with the per-sample contiguous row layout the model expects
    (batch, half, symbol, rx - see train_chanest._pilot_rows()) only when
    B == 1. This applies the same regrouping the training dataset uses, so
    batched inference feeds the model exactly the same tensors as B=1 calls.
    """
    R, one, L, two = x.shape
    x = x.reshape(n_half, n_symbol, B, n_rx, one, L, two)
    x = x.permute(2, 0, 1, 3, 4, 5, 6)
    return x.reshape(R, one, L, two)


def _ungroup_rows_per_sample(x, B, n_half=2, n_symbol=2, n_rx=2):
    """Inverse of _regroup_rows_per_sample:
    (batch, half, symbol, rx) -> (half, symbol, batch, rx)."""
    R, one, L, two = x.shape
    x = x.reshape(B, n_half, n_symbol, n_rx, one, L, two)
    x = x.permute(1, 2, 0, 3, 4, 5, 6)
    return x.reshape(R, one, L, two)


def inference_chanest(model, DMRS_rcv, DMRS_pilot, device='cpu',
                     nRx=2, nSymbol=2):
    B = DMRS_rcv.shape[0]
    pilot, received_DMRS = preprocess_chanest(DMRS_rcv, DMRS_pilot)
    if B > 1:
        # regroup rows to the model's per-sample layout (batch, half, symbol, rx)
        pilot = _regroup_rows_per_sample(pilot, B, 2, nSymbol, nRx)
        received_DMRS = _regroup_rows_per_sample(received_DMRS, B, 2, nSymbol, nRx)
    start_time = time.perf_counter()
    output_hat, DMRS_est, SNR_est = infer_chanest(model, pilot, received_DMRS, device)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Running time: {execution_time:.4f} s")
    if B > 1:
        # model rows come back as (batch, half, symbol, rx); postprocess_chanest()
        # expects the preprocess order (half, symbol, batch, rx)
        DMRS_est = _ungroup_rows_per_sample(DMRS_est, B, 2, nSymbol, nRx)
    H_est_complex, noise_var_est = postprocess_chanest(DMRS_est, SNR_est, nRx, nSymbol)
    return H_est_complex, noise_var_est



def preprocess_pocdet(DMRS_ch, Sig_rcv, noise_var=None):

    nRB = 64
    nRE = 12

    B, nRx, nSymbol, nRBRE = DMRS_ch.shape
    DMRS_ch_np = np.zeros((B, nRx, nSymbol, nRBRE, 2))
    DMRS_ch_np[..., 0] = DMRS_ch.real
    DMRS_ch_np[..., 1] = DMRS_ch.imag
    DMRS_ch_np = np.reshape(DMRS_ch_np, [B, nRx, nSymbol, 2, nRB // 2, nRE, 2])
    DMRS_ch_np = np.transpose(DMRS_ch_np, [3, 2, 0, 1, 4, 5, 6])
    DMRS_ch_np = np.reshape(DMRS_ch_np, [2 * nSymbol * B, nRx, nRB // 2 * nRE, 2])

    B, nRx, nSymbol, nRBRE = Sig_rcv.shape
    Sig_rcv_np = np.zeros((B, nRx, nSymbol, nRBRE, 2))
    Sig_rcv_np[..., 0] = Sig_rcv.real
    Sig_rcv_np[..., 1] = Sig_rcv.imag
    Sig_rcv_np = np.reshape(Sig_rcv_np, [B, nRx, nSymbol, 2, nRB // 2, nRE, 2])
    Sig_rcv_np = np.transpose(Sig_rcv_np, [3, 2, 0, 1, 4, 5, 6])
    Sig_rcv_np = np.reshape(Sig_rcv_np, [2 * nSymbol * B, nRx, nRB // 2 * nRE, 2])

    noise_var_np = np.array(noise_var)
    noise_var_np = np.repeat(noise_var_np[:, :, np.newaxis], 2 * nSymbol, axis=2)
    noise_var_np = np.transpose(noise_var_np, [2, 1, 0])
    noise_var = np.reshape(noise_var_np, [2 * nSymbol * B, 1])

    channel = torch.from_numpy(DMRS_ch_np).float()
    received_data = torch.from_numpy(Sig_rcv_np).float()
    noise_var = torch.from_numpy(noise_var).float()

    return channel, received_data, noise_var


def infer_pocdet(model, channel, received_data, noise_var, device='cpu'):

    channel = channel.to(device)
    received_data = received_data.to(device)
    noise_var = noise_var.to(device)

    with torch.no_grad():
        output_hat, data_hat = model(
            channel=channel,
            received_data=received_data,
            noise_power=noise_var,
            instruction=None,
            ta_perform='pocdet'
        )

    return output_hat, data_hat


def postprocess_pocdet(data_hat, nSymbol=12):

    nRB = 64
    nRE = 12

    data_hat_np = data_hat.cpu().numpy()
    total_B = data_hat_np.shape[0]
    B = total_B // (2 * nSymbol)


    data_hat_np = np.reshape(data_hat_np, [2, nSymbol, B, 1, nRB // 2, nRE, 2])

    data_hat_np = np.transpose(data_hat_np, [2, 3, 1, 0, 4, 5, 6])

    data_hat_np = np.reshape(data_hat_np, [B, nSymbol, nRB * nRE, 2])


    Sig_eq_complex = data_hat_np[..., 0] + 1j * data_hat_np[..., 1]
    Sig_eq_max = 0.05
    Sig_eq_complex = Sig_eq_complex * Sig_eq_max

    return Sig_eq_complex


def inference_pocdet(model, DMRS_ch, Sig_rcv, noise_var=None, device='cpu',
                     nSymbol=12):

    channel, received_data, noise_var = preprocess_pocdet(DMRS_ch, Sig_rcv, noise_var)
    output_hat, data_hat = infer_pocdet(model, channel, received_data, noise_var, device)
    Sig_eq_complex = postprocess_pocdet(data_hat, nSymbol)
    return Sig_eq_complex




def load_chanest_data(mat_path):
    ##old
    # mat_data = scipy.io.loadmat(mat_path)
    # struct = mat_data['StateOut4Est'][0, 0]
    # DMRS_rcv = struct['rxDmRsRe']
    # DMRS_pilot = struct['dmrsTx']
    # noise_var_ideal = struct['nvar_ideal']
    # noise_var_est = struct['nvar_est']
    ##new
    with h5py.File(mat_path, 'r') as f:
        struct_group = f['StateOut4Est']

        # Read complex data from HDF5 compound type (MATLAB v7.3 format)
        def read_complex(ds):
            data = ds[()]
            if data.dtype.names:  # compound type for complex numbers
                return (data['real'] + 1j * data['imag']).T
            return np.array(data).T

        DMRS_rcv = read_complex(struct_group['rxDmRsRe'])  # (B, nRx, nSymbol, nRBRE) complex128
        DMRS_pilot = read_complex(struct_group['dmrsTx'])  # (B, nRx, nSymbol, nRBRE) complex128
        DMRS_ch_true = read_complex(struct_group['dmrsH'])  # (B, nRx, nSymbol, nRBRE) complex128
        noise_var_ideal = np.array(struct_group['nvar_ideal']).T  # (1, B) float64
        noise_var_est = np.array(struct_group['nvar_est']).T  # (1, B) float64
        
        ##remain satisfied SNR
#        noise_type = np.unique(noise_var_ideal)
#        [index_N] = np.where(noise_var_ideal[0,:] == noise_type[0])
#        DMRS_rcv = DMRS_rcv[index_N,:,:,:]
#        DMRS_pilot = DMRS_pilot[index_N,:,:,:]
#        DMRS_ch_true = DMRS_ch_true[index_N,:,:,:]
#        noise_var_ideal = noise_var_ideal[:,index_N]
#        noise_var_est = noise_var_est[:,index_N]
    return DMRS_rcv, DMRS_pilot, DMRS_ch_true , noise_var_ideal, noise_var_est


def load_pocdet_data(mat_path):
    # mat_data = scipy.io.loadmat(mat_path)
    # struct = mat_data['StateOut4Eq'][0, 0]
    # DMRS_ch = struct['H']
    # Sig_rcv = struct['rxPuschRe']
    # noise_var_ideal = struct['nvar_ideal']
    # noise_var_est = struct['nvar_est']
    # Sig_eq = struct['txSigFreq_noDmrs']

    ##new
    with h5py.File(mat_path, 'r') as f:
        struct_group = f['StateOut4Eq']

        # Read complex data from HDF5 compound type (MATLAB v7.3 format)
        def read_complex(ds):
            data = ds[()]
            if data.dtype.names:  # compound type for complex numbers
                return (data['real'] + 1j * data['imag']).T
            return np.array(data).T

        DMRS_ch = read_complex(struct_group['H'])
        Sig_rcv = read_complex(struct_group['rxPuschRe'])
        noise_var_ideal = np.array(struct_group['nvar_ideal']).T
        noise_var_est = np.array(struct_group['nvar_est']).T
        Sig_eq = read_complex(struct_group['txSigFreq_noDmrs'])

    
    ##remain satisfied SNR
    noise_type = np.unique(noise_var_ideal)
    [index_N] = np.where(noise_var_ideal[0,:] == noise_type[0])
    DMRS_ch = DMRS_ch[index_N,:,:,:]
    Sig_rcv = Sig_rcv[index_N, :, :,:]
    Sig_eq = Sig_eq[index_N,:,:]
    noise_var_ideal = noise_var_ideal[:,index_N]
    noise_var_est = noise_var_est[:,index_N]
    ##remove zero
    nRB = 64
    nRE = 12
    Sig_eq[:, 0, int(nRB * nRE / 2):] = Sig_eq[:, 10, 0:int(nRB * nRE / 2)]
    Sig_eq = np.delete(Sig_eq, 10, axis=1)
    DMRS_ch[:, :, 0, int(nRB * nRE / 2):] = DMRS_ch[:, :, 10, 0:int(nRB * nRE / 2)]
    DMRS_ch = np.delete(DMRS_ch, 10, axis=2)
    Sig_rcv[:, :, 0, int(nRB * nRE / 2):] = Sig_rcv[:, :, 10, 0:int(nRB * nRE / 2)]
    Sig_rcv = np.delete(Sig_rcv, 10, axis=2)

    return DMRS_ch, Sig_rcv, noise_var_ideal, noise_var_est,Sig_eq

def NMSE(x_hat, x):
    power = np.sum(np.abs(x ** 2))
    mse = np.sum(np.abs((x - x_hat) ** 2))
    nmse = mse / power
    return nmse

# ================================================================
#                    Main
# ================================================================

if __name__ == "__main__":

    # ================= batch inference setting =================
    # number of samples processed per model call. The original scripts
    # always ran one sample per call; batch_num = 1 reproduces exactly
    # the original behaviour, larger values amortize the per-call
    # overhead and use the GPU more efficiently.
    batch_num = 128

    device = torch.device('cuda:0')
    # no-op on CPU; on GPU, torch.cuda.synchronize makes perf_counter
    # measure the real (asynchronous) kernel execution time
    sync = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)
    # load model
    model_path = "./multi-task-LLM/output/WLMtra.bin"  #
    model = load_model(model_path, device)


    # ============= channel estimation =============#


    ce_mat_path = "./multi-task-LLM/data_CE/Train/Chest_USRP_Octv1.mat"

    DMRS_rcv, DMRS_pilot, DMRS_ch_true,noise_var_ideal, noise_var_est = load_chanest_data(ce_mat_path)
    print(f"DMRS_rcv shape: {DMRS_rcv.shape}, dtype: {DMRS_rcv.dtype}")
    print(f"DMRS_pilot shape: {DMRS_pilot.shape}, dtype: {DMRS_pilot.dtype}")
    DMRS_ch_est = np.zeros_like(DMRS_ch_true)
    noise_var_ai = np.zeros_like(noise_var_ideal)

    ## batched inference (batch_num samples per model call)
    num_sample = np.size(DMRS_ch_true,axis=0)
    NMSE_N = np.zeros(7) ##noise level 1 1.58, 2.5, 3.9, 6.3, 10
    NMSE_H = np.zeros(7)
    num_count = np.zeros(7)
    noise_type = np.unique(noise_var_ideal)

    # per-phase timing (preprocess / model forward / postprocess):
    # the two numpy phases are identical for every model, so only the
    # "fwd" column reveals the model speedup. NOTE: a bare perf_counter
    # around the async model call measures only the CPU launch time -
    # every phase below is bracketed with torch.cuda.synchronize().
    time_pre = time_fwd = time_post = 0.0
    for i in range(0, num_sample, batch_num):
        e = min(i + batch_num, num_sample)
        # -- phase 1: numpy preprocessing + per-sample row regroup -----
        sync()
        t0 = time.perf_counter()
        pilot, received_DMRS = preprocess_chanest(
            DMRS_rcv[i:e,:,:,:], DMRS_pilot[i:e,:,:,:])
        pilot = _regroup_rows_per_sample(pilot, e - i, 2, 2, 2)
        received_DMRS = _regroup_rows_per_sample(received_DMRS, e - i, 2, 2, 2)
        sync()
        time_pre += time.perf_counter() - t0
        # -- phase 2: the model forward (H2D + GPU kernels) -----------
        sync()
        t0 = time.perf_counter()
        output_hat, DMRS_est, Noise_power_est = infer_chanest(
            model, pilot, received_DMRS, device)
        sync()
        time_fwd += time.perf_counter() - t0
        # -- phase 3: row ungroup + numpy postprocessing --------------
        sync()
        t0 = time.perf_counter()
        DMRS_est = _ungroup_rows_per_sample(DMRS_est, e - i, 2, 2, 2)
        H_est_complex, noise_var_est_out = postprocess_chanest(
            DMRS_est, Noise_power_est)
        sync()
        time_post += time.perf_counter() - t0
        noise_var_nvar = compute_noise_variance_nvar(H_est_complex, DMRS_rcv[i:e, :, :, :], DMRS_pilot[i:e, :, :, :], nRx=2)
        DMRS_ch_est[i:e,:,:,:] = H_est_complex
        noise_var_ai[0, i:e] = noise_var_nvar
        ##per-sample metrics (identical to the original per-sample loop)
        for j in range(i, e):
            NMSE_H_cur = NMSE(np.reshape(DMRS_ch_true[j:j+1,:,:,:],(-1,1)), np.reshape(H_est_complex[j-i:j-i+1,:,:,:],(-1,1)))
            NMSE_N_cur = NMSE(noise_var_ideal[0:1,j] ,noise_var_est_out[:, j-i:j-i+1])
            NMSE_H[0] += NMSE_H_cur
            NMSE_N[0] += NMSE_N_cur
            num_count[0] += 1
            #[index_N] = np.where(noise_type == noise_var_ideal[0,j])
            #NMSE_H[index_N+1] += NMSE_H_cur
            #NMSE_N[index_N + 1] += NMSE_N_cur
            #num_count[index_N+1] += 1
        if e % 5==0 or e == num_sample:
            time_ce = time_pre + time_fwd + time_post
            print(f"Channel_NMSE: {NMSE_H/num_count}, Noise_NMSE: {NMSE_N/num_count}, Avg_time: {time_ce/e*1000:.3f} ms/sample (pre {time_pre/e*1000:.3f} / fwd {time_fwd/e*1000:.3f} / post {time_post/e*1000:.3f})")
    NMSE_H_mean = NMSE_H/num_count
    NMSE_N_mean = NMSE_N/num_count
    savemat('CE_result.mat', {'Channel_NMSE': NMSE_H_mean, 'Noise_NMSE': NMSE_N_mean})
    savemat('CE_output.mat', {'DMRS_est': DMRS_ch_est,'Noise_est': noise_var_ai})
    time_ce = time_pre + time_fwd + time_post
    print(f"Channel_NMSE: {NMSE_H_mean}, Noise_NMSE: {NMSE_N_mean}, Avg_time: {time_ce/num_sample*1000:.3f} ms/sample (pre {time_pre/num_sample*1000:.3f} / fwd {time_fwd/num_sample*1000:.3f} / post {time_post/num_sample*1000:.3f})")


    # ============= PoC detection=============#


    dec_mat_path = "./multi-task-LLM/data_DEC/Train/Cheq_PoC16.mat"

    # load data
    DMRS_ch, Sig_rcv, noise_var_ideal, noise_var_est,Sig_eq_true = load_pocdet_data(dec_mat_path)
    Sig_eq_est = np.zeros_like(Sig_eq_true)


    # Full inference: Input DMRS channel, received signal, and noise variance, output detected data
    # Sig_eq_complex = inference_pocdet(
    #     model, DMRS_ch, Sig_rcv, noise_var_ideal, device=device
    # )

    ## batched inference (batch_num samples per model call)
    num_sample = np.size(Sig_rcv,axis=0)
    NMSE_Sig = np.zeros(7)
    num_count = np.zeros(7)
    noise_type = np.unique(noise_var_ideal) ##noise level 1.58, 2.5, 3.9, 6.3, 10
    time_det = 0.0   # accumulated PoC-detection inference time (s)
    for i in range(0, num_sample, batch_num):
        e = min(i + batch_num, num_sample)
        # Full inference: Input DMRS channel, received signal, and noise variance, output detected data
        sync()
        t0 = time.perf_counter()
        Sig_eq_complex = inference_pocdet(
            model, DMRS_ch[i:e,:,:,:], Sig_rcv[i:e,:,:,:], noise_var_ideal[:,i:e], device=device
        )
        sync()
        time_det += time.perf_counter() - t0
        Sig_eq_est[i:e,:,:] = Sig_eq_complex
        ##data process (per-sample metrics, identical to the original loop)
        for j in range(i, e):
            NMSE_Sig_cur = NMSE(np.reshape(Sig_eq_true[j:j+1,:,:],(-1,1)), np.reshape(Sig_eq_complex[j-i:j-i+1,:,:],(-1,1)))
            NMSE_Sig[0] += NMSE_Sig_cur
            num_count[0] += 1
            [index_N] = np.where(noise_type == noise_var_ideal[0,j])
            NMSE_Sig[index_N+1] += NMSE_Sig_cur
            num_count[index_N+1] += 1
        if e % 100==0 or e == num_sample:
            print(f"Signal_NMSE: {NMSE_Sig/num_count}, Avg_time: {time_det/e*1000:.3f} ms/sample")
    NMSE_Sig_mean = NMSE_Sig/num_count
    savemat('DET_result.mat', {'Signal_NMSE': NMSE_Sig_mean})
    savemat('DET_output.mat', {'Sig_eq_est': Sig_eq_est})
    print(f"Signal_NMSE: {NMSE_Sig_mean}, Avg_time: {time_det/num_sample*1000:.3f} ms/sample")

