"""Block-8 timing phase: what planning and data encoding cost on the job's GPU.

    python -m cvar.eval.gpu_timing --out R/timing [--device cuda] [--smoke]

Writes R/timing/timing.json (atomic; a rerun skips if it exists). Measures:
  1. model.get_cost at candidate batch N in {300, 1200, 2400, 4800} = 300 candidates x S futures,
     S in {1, 4, 8, 16} (candidates repeated through the deterministic predictor as a stand-in for S
     samples). One get_cost call = one CEM iteration; median of 10 calls after 3 warm-up calls,
     cuda-synchronized; peak memory per N. est_plan_call_s = n_steps (30) x median.
  2. Encoder throughput (frames/s) at batch 256, fp32, median of 10 after warm-up.
  3. Collect-and-encode in one pass: stochastic PushT (p = 0.15, medium kicks) with WeakPolicy-style
     near-block actions, rendering 224 px frames on the CPU, frames batched (256) to the encoder on
     the device (uint8 upload, ImageNet normalization on the device), latents back as float16.
     Reports steps/s overall and the env / encode split; checks the on-device preprocessing against
     eval.py's CPU transform (max |emb diff|).
  4. Env-only render throughput with CVL_NCPU worker processes (the ceiling for a parallel collector).
  5. fp16 storage check on the collected latents: max |z - fp32(fp16(z))| and the ratio of the fp16
     rounding MSE to the MSE between consecutive steps' latents (decision 2026-11-13: fp32 if > 1e-2).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device: str, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    return ts


def get_cost_timing(model, device, sizes, warmup, reps, n_steps) -> list[dict]:
    from cvar.planning.bench_cpu import eval_transform, load_frames

    tf = eval_transform()
    f0, fg, _ = load_frames()
    base = {"pixels": tf(f0)[None, None].to(device), "goal": tf(fg)[None, None].to(device),
            "action": torch.zeros(1, 1, 10, device=device)}
    g = torch.Generator(device=device).manual_seed(0)
    out = []
    for n in sizes:
        info = {k: v.unsqueeze(1).expand(1, n, *v.shape[1:]) for k, v in base.items()}  # as CEMSolver
        cand = torch.randn(1, n, 5, 10, generator=g, device=device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                ts = timed(lambda: model.get_cost(dict(info), cand), device, warmup, reps)
            med = statistics.median(ts)
            row = {"N": n, "S": n // 300 if n >= 300 else None, "median_s": med, "times_s": ts,
                   "est_plan_call_s": n_steps * med}
            if device.startswith("cuda"):
                row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        except torch.cuda.OutOfMemoryError as e:  # record and keep going
            row = {"N": n, "error": f"OOM: {e}"[:300]}
            torch.cuda.empty_cache()
        print("get_cost", {k: v for k, v in row.items() if k != "times_s"}, flush=True)
        out.append(row)
    return out


def device_preprocess(device: str):
    import stable_pretraining as spt

    mean = torch.tensor(spt.data.dataset_stats.ImageNet["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(spt.data.dataset_stats.ImageNet["std"], device=device).view(1, 3, 1, 1)

    def pre(frames_u8: np.ndarray) -> torch.Tensor:  # (B, H, W, 3) uint8 -> (B, 1, 3, 224, 224)
        x = torch.from_numpy(frames_u8).to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        assert x.shape[-1] == 224 and x.shape[-2] == 224, "frames must already be 224 px (Resize is then a no-op)"
        return ((x - mean) / std)[:, None]
    return pre


def collect_and_encode(model, device: str, n_steps: int, batch: int) -> tuple[dict, np.ndarray, np.ndarray]:
    import stable_worldmodel.envs.pusht.env as pusht
    from cvar.data.throughput_probe import near_block_action
    from cvar.envs.stochastic_pusht import StochasticPushT
    from cvar.planning.bench_cpu import eval_transform

    pre = device_preprocess(device)
    env = StochasticPushT(pusht.PushT(), p=0.15)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    buf, lat, ep_ids, ep = [], [], [], 0
    t_env = t_enc = 0.0
    check = None
    t_all = time.perf_counter()
    for t in range(n_steps):
        t0 = time.perf_counter()
        _, _, term, trunc, _ = env.step(near_block_action(env.unwrapped, rng))
        buf.append(env.render())
        ep_ids.append(ep)
        if term or trunc or (t + 1) % 150 == 0:
            ep += 1
            env.reset(seed=ep)
        t_env += time.perf_counter() - t0
        if len(buf) == batch or t == n_steps - 1:
            t0 = time.perf_counter()
            frames = np.stack(buf)
            with torch.inference_mode():
                z = model.encode({"pixels": pre(frames)})["emb"][:, -1]
                lat.append(z.float().cpu())
            sync(device)
            t_enc += time.perf_counter() - t0
            if check is None:  # on-device preprocessing vs eval.py's CPU transform, first batch
                tf = eval_transform()
                with torch.inference_mode():
                    ref = model.encode({"pixels": torch.stack([tf(f) for f in frames[:16]])[:, None].to(device)})
                check = float((ref["emb"][:, -1].float().cpu() - lat[-1][:16]).abs().max())
            buf = []
    wall = time.perf_counter() - t_all
    z = torch.cat(lat).numpy()
    res = {"n_steps": n_steps, "batch": batch, "wall_s": wall, "steps_per_s": n_steps / wall,
           "env_render_s": t_env, "encode_s": t_enc, "env_steps_per_s": n_steps / t_env,
           "encode_frames_per_s_in_loop": n_steps / t_enc, "device_preprocess_max_abs_emb_diff": check}
    return res, z, np.array(ep_ids)


def fp16_check(z: np.ndarray, ep: np.ndarray) -> dict:
    z16 = z.astype(np.float16).astype(np.float32)
    rmse = float(((z - z16) ** 2).mean())
    same = ep[1:] == ep[:-1]
    step = float(((z[1:][same] - z[:-1][same]) ** 2).mean())
    return {"n": int(len(z)), "abs_max_latent": float(np.abs(z).max()), "max_abs_round_err": float(np.abs(z - z16).max()),
            "round_mse": rmse, "consecutive_step_mse": step, "ratio_round_to_step": rmse / step,
            "decision_rule": "store fp16 unless ratio > 1e-2"}


def _env_worker(args):
    from cvar.data.throughput_probe import env_rate
    torch.set_num_threads(1)
    rate, _ = env_rate(True, args[0], seed=args[1])
    return rate


def parallel_env_rate(n_proc: int, steps_each: int) -> dict:
    ctx = mp.get_context("fork")
    t0 = time.perf_counter()
    with ctx.Pool(n_proc) as pool:
        rates = pool.map(_env_worker, [(steps_each, 100 * i) for i in range(n_proc)])
    wall = time.perf_counter() - t0
    return {"processes": n_proc, "steps_each": steps_each, "per_process_steps_per_s": rates,
            "aggregate_steps_per_s": n_proc * steps_each / wall, "wall_s": wall}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if (out / "timing.json").exists() and not a.force:
        print(f"timing: {out / 'timing.json'} exists; skipping (resume)", flush=True)
        return 0
    ncpu = int(os.environ.get("CVL_NCPU", "1"))
    torch.set_num_threads(ncpu)
    from cvar.utils.checkpoint import load_lewm

    t0 = time.time()
    model = load_lewm("pusht/lewm", a.device)
    model.interpolate_pos_encoding = True
    res = {"device": a.device, "torch": torch.__version__, "smoke": a.smoke, "cvl_ncpu": ncpu}
    if a.device.startswith("cuda"):
        p = torch.cuda.get_device_properties(0)
        res["gpu"] = {"name": p.name, "capability": f"{p.major}.{p.minor}", "mem_gb": p.total_memory / 2**30,
                      "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    sizes = [30, 120] if a.smoke else [300, 1200, 2400, 4800]
    res["get_cost"] = get_cost_timing(model, a.device, sizes, warmup=1 if a.smoke else 3,
                                      reps=2 if a.smoke else 10, n_steps=30)

    from cvar.data.throughput_probe import env_rate  # noqa: F401  (import check before the long part)
    b = 16 if a.smoke else 256
    _, z_frames, _ = collect_and_encode(model, a.device, n_steps=b, batch=b)  # warm-up pass
    frames = np.zeros((b, 224, 224, 3), np.uint8)
    pre = device_preprocess(a.device)
    with torch.inference_mode():
        ts = timed(lambda: model.encode({"pixels": pre(frames)}), a.device, 2, 3 if a.smoke else 10)
    res["encode"] = {"batch": b, "median_s": statistics.median(ts), "frames_per_s": b / statistics.median(ts)}
    print("encode", res["encode"], flush=True)

    ce, z, ep = collect_and_encode(model, a.device, n_steps=64 if a.smoke else 4000, batch=b)
    res["collect_and_encode"] = ce
    print("collect_and_encode", ce, flush=True)
    res["fp16"] = fp16_check(z, ep)
    print("fp16", res["fp16"], flush=True)
    res["parallel_env"] = parallel_env_rate(ncpu, 100 if a.smoke else 1500)
    print("parallel_env", res["parallel_env"], flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "timing.json.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(out / "timing.json")
    print("TIMING_OK", out / "timing.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
