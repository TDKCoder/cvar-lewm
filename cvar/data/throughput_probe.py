"""Phase 0 block 5: what data collection and pre-encoding cost in the CPU worker.

    python -m cvar.data.throughput_probe --out $RUNS_DIR/phase0-data-probe

Measures (single process, CVL_NCPU torch threads):
  - stochastic PushT (p = 0.15) env steps/s with WeakPolicy-style near-block actions, with and
    without render() of the 224x224 frame;
  - frozen LeWM encoder throughput on CPU (frames/s) at batch 64, median of 5 after warm-up;
  - bytes per stored frame: raw 224 / 96 / 64 px uint8, PNG and JPEG (q 90) at those sizes, and
    np.savez_compressed of 64 px stacks; latent bytes (192 x fp16 / fp32).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch


def near_block_action(base, rng):
    a = rng.uniform(-1, 1, 2) * base.action_scale + np.array(base.agent.position)
    bp = np.array(base.block.position)
    a = np.clip(a, bp - 60, bp + 60)
    return np.clip((a - np.array(base.agent.position)) / base.action_scale, -1, 1).astype(np.float32)


def env_rate(render: bool, n: int, seed: int = 0):
    import stable_worldmodel.envs.pusht.env as pusht
    from cvar.envs.stochastic_pusht import StochasticPushT

    env = StochasticPushT(pusht.PushT(), p=0.15)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    frames, steps, t0 = [], 0, time.perf_counter()
    while steps < n:
        _, _, term, trunc, _ = env.step(near_block_action(env.unwrapped, rng))
        if render:
            f = env.render()
            if len(frames) < 200 and steps % 3 == 0:
                frames.append(f.copy())
        steps += 1
        if term or trunc or steps % 150 == 0:
            env.reset(seed=seed + steps)
    return n / (time.perf_counter() - t0), frames


def frame_bytes(frames: list[np.ndarray]) -> dict:
    out = {}
    for px in (224, 96, 64):
        fs = [f if px == 224 else cv2.resize(f, (px, px), interpolation=cv2.INTER_AREA) for f in frames]
        out[f"raw_{px}"] = int(fs[0].nbytes)
        out[f"png_{px}"] = float(np.mean([len(cv2.imencode(".png", f)[1]) for f in fs]))
        out[f"jpeg90_{px}"] = float(np.mean([len(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]) for f in fs]))
        out[f"zlib6_{px}"] = len(zlib.compress(np.stack(fs).tobytes(), 6)) / len(fs)
    out["latent_fp16"], out["latent_fp32"] = 192 * 2, 192 * 4
    return out


def encode_rate(frames, batch=64, repeats=5):
    from cvar.planning.bench_cpu import eval_transform
    from cvar.utils.checkpoint import load_lewm

    model = load_lewm("pusht/lewm", "cpu")
    tf = eval_transform()
    x = torch.stack([tf(frames[i % len(frames)]) for i in range(batch)])[:, None]  # (B,1,3,224,224)
    with torch.inference_mode():
        model.encode({"pixels": x.clone()})
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.encode({"pixels": x.clone()})
            ts.append(time.perf_counter() - t0)
    return batch / statistics.median(ts), ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ["CVL_NCPU"]))
    res = {"threads": int(os.environ["CVL_NCPU"]), "torch": torch.__version__}
    res["env_steps_per_s_no_render"], _ = env_rate(False, a.steps)
    res["env_steps_per_s_render224"], frames = env_rate(True, a.steps)
    print({k: round(v, 1) for k, v in res.items() if k.startswith("env")}, flush=True)
    res["bytes_per_frame"] = frame_bytes(frames)
    print(res["bytes_per_frame"], flush=True)
    res["encode_frames_per_s_cpu_b64"], res["encode_batch_s"] = encode_rate(frames)
    print("encode frames/s (CPU, batch 64):", round(res["encode_frames_per_s_cpu_b64"], 1), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "probe.json").write_text(json.dumps(res, indent=1))
    print("PROBE_OK")


if __name__ == "__main__":
    main()
