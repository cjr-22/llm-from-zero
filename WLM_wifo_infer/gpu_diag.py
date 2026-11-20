# -*- coding: utf-8 -*-
"""GPU timing diagnostic for the WiFo2 chanest inference scripts.

Run this ON THE GPU MACHINE, from the WLM_wifo_infer directory:

    python gpu_diag.py

It loads both models (base + enhance), replays the exact per-sample
inference flow of the two scripts, and prints a per-component timing
breakdown so we can see where the hundreds of milliseconds go.
"""
import importlib.util
import os
import time

import numpy as np
import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("=" * 72)
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("CUDA_LAUNCH_BLOCKING =", os.environ.get("CUDA_LAUNCH_BLOCKING"))
print("num processes hint -> run `nvidia-smi` separately to check GPU load")
print("=" * 72)

base = load_module("inference_GPUfixedSNR_noise_Oct_WiFo2.py", "base_inf")
enh = load_module("inference_GPUfixedSNR_noise_Oct_WiFo2_enhance.py", "enh_inf")
device = torch.device("cuda:0")

rng = np.random.default_rng(0)


def make_data(B):
    RX = rng.standard_normal((B, 2, 2, 384)) + 1j * rng.standard_normal((B, 2, 2, 384))
    TX = rng.standard_normal((B, 1, 2, 384)) + 1j * rng.standard_normal((B, 1, 2, 384))
    return RX, TX


def med(v):
    v = sorted(v)
    return v[len(v) // 2]


def bench(mod, name, ckpt, B, n=15):
    """Replay the per-sample flow of the script and time each component."""
    RX, TX = make_data(B)
    model = mod.load_model(ckpt, device)

    comp = {k: [] for k in ("preprocess", "h2d", "fwd_enqueue", "fwd_sync",
                            "pure_fwd", "postprocess", "nvar", "total")}
    pilot_g = received_g = None
    for it in range(n):
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        pilot, received = mod.preprocess_chanest(RX, TX)
        comp["preprocess"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pilot_g = pilot.to(device)
        received_g = received.to(device)
        torch.cuda.synchronize()
        comp["h2d"].append(time.perf_counter() - t0)

        # the script's own measurement style (enqueue only, no sync)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(pilot_sequence=pilot_g, received_DMRS=received_g,
                    instruction=None, ta_perform="chanest")
        comp["fwd_enqueue"].append(time.perf_counter() - t0)
        torch.cuda.synchronize()
        comp["fwd_sync"].append(time.perf_counter() - t0)

        # pure forward on already-resident tensors (like infer_chanest does)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(pilot_sequence=pilot.to(device),
                  received_DMRS=received.to(device),
                  instruction=None, ta_perform="chanest")
        torch.cuda.synchronize()
        comp["pure_fwd"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        mod.postprocess_chanest(out[1], out[2], 2, 2)
        comp["postprocess"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        H, _ = mod.postprocess_chanest(out[1], out[2], 2, 2)
        mod.compute_noise_variance_nvar(H, RX, TX, nRx=2)
        comp["nvar"].append(time.perf_counter() - t0)

        torch.cuda.synchronize()
        comp["total"].append(time.perf_counter() - t_start)

    print(f"\n--- {name} (B={B}, median of {n}) ---")
    for k in ("preprocess", "h2d", "fwd_enqueue", "fwd_sync", "pure_fwd",
              "postprocess", "nvar", "total"):
        print(f"  {k:12s}: {med(comp[k]) * 1000:8.2f} ms")


if __name__ == "__main__":
    for B in (1, 8):
        bench(base, "base  ", "./multi-task-LLM/output/WLMtra.bin", B)
        bench(enh, "enhance", "./multi-task-LLM/output/enhance/Noagg/WLMtra.bin", B)
