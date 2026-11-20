# -*- coding: utf-8 -*-
"""enhance_batch (Noagg, tau=0)  vs  agg_batch (tau=2, 0.7/0.7,
single-sync compaction) at batch_num = 32 / 128, on the real
checkpoints and the first 512 samples of the CE dataset.

The report is printed AND saved to exp_compare_result.txt.

Run:  python exp_compare.py
"""
import datetime
import importlib.util
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

RESULT_FILE = "exp_compare_result.txt"
LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))


log(f"exp_compare report  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
log(f"torch {torch.__version__}, device available: "
    f"cuda={torch.cuda.is_available()}"
    + (f" ({torch.cuda.get_device_name(0)})"
       if torch.cuda.is_available() else ""))

spec = importlib.util.spec_from_file_location(
    "inf", "inference_GPUfixedSNR_noise_Oct_WiFo2_enhance_batch.py")
inf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inf)

from train_chanest_consis_auto import load_chanest_model_auto

DATA = "./multi-task-LLM/data_CE/Train/Chest_USRP_Octv1.mat"
N_SUB = 512


def load_subset(mat_path, n):
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


log("loading data ...")
DMRS_rcv, DMRS_pilot, DMRS_ch_true, nv_ideal, nv_est = load_subset(DATA, N_SUB)
n_all = DMRS_rcv.shape[0]
log(f"samples: {n_all}")

log("loading enhance (Noagg) model ...")
m_no = load_chanest_model_auto(
    "./multi-task-LLM/output/enhance/Noagg/WLMtra.bin", "cpu")
m_no.eval()          # tau_eval = 0 (no config.json) -> no merging

log("loading agg model ...")
m_ag = load_chanest_model_auto(
    "./multi-task-LLM/output/enhance/Agg/WLMtra.bin", "cpu")
m_ag.eval()
m_ag.tau_eval = 2
m_ag.theta_c = 0.7
m_ag.theta_k = 0.7
m_ag.ata_static = True          # single-sync compaction mode
m_ag.ata_final_compact = True


def run(model, batch_num, n_batches, offset):
    t_sum, nmse_h, nmse_n, cnt = 0.0, 0.0, 0.0, 0
    ratios = []
    for b in range(n_batches):
        i = offset + b * batch_num
        e = min(i + batch_num, n_all)
        pilot, recv = inf.preprocess_chanest(
            DMRS_rcv[i:e], DMRS_pilot[i:e])
        t0 = time.perf_counter()
        out, dmrs, nvar = inf.infer_chanest(model, pilot, recv, "cpu")
        t_sum += time.perf_counter() - t0
        ratios.append(model.last_token_ratio)
        H, nv = inf.postprocess_chanest(dmrs, nvar)
        nmse_h += inf.NMSE(np.reshape(DMRS_ch_true[i:e], (-1, 1)),
                           np.reshape(H, (-1, 1)))
        nmse_n += inf.NMSE(nv_ideal[0:1, i:e], nv)
        cnt += e - i
    return (t_sum / cnt * 1000.0, t_sum / n_batches * 1000.0,
            float(np.mean(ratios)), nmse_h / n_batches,
            nmse_n / n_batches)


for batch_num in (32, 128):
    log(f"\n===== batch_num = {batch_num} "
        f"(warmup 1 batch + 3 timed batches) =====")
    log(f"{'model':<38}{'ms/sample':>11}{'ms/batch':>10}"
        f"{'ratio':>8}{'Ch_NMSE':>11}{'N_NMSE':>11}")
    log("-" * 89)
    base = None
    for name, model in [("enhance (Noagg, tau=0)", m_no),
                        ("agg (tau=2, 0.7/0.7, single-sync)", m_ag)]:
        run(model, batch_num, n_batches=1, offset=0)        # warmup
        ms, ms_b, ratio, h, nv = run(model, batch_num, n_batches=3,
                                     offset=batch_num)
        if base is None:
            base = ms
        log(f"{name:<38}{ms:>11.2f}{ms_b:>10.1f}{ratio:>8.3f}"
            f"{h:>11.4g}{nv:>11.4g}"
            f"   speedup {base / ms:>5.2f}x")

with open(RESULT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(LOG) + "\n")
print(f"\nreport saved to {RESULT_FILE}")
