# SPDX-License-Identifier: Apache-2.0
"""Deterministic MiniMax H3 t2va A/B probe.

Runs fixed-seed t2va generations on a single GPU and writes SHA256 hashes +
stats + per-run denoise wall-clock to JSON, so the same run before and after the
token-refiner-cache change can be compared bit-for-bit and timed. One warmup run
is discarded; the reported time is the median of the measured runs.
"""

import argparse
import hashlib
import json
import os
import statistics
import time

import numpy as np
import torch


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--offload", choices=["model", "layerwise", "none"], default="model")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    model = os.environ["MODEL"]
    if args.offload == "layerwise":
        offload_kwargs = {"enable_layerwise_offload": True}
    elif args.offload == "model":
        offload_kwargs = {"enable_cpu_offload": True}
    else:
        offload_kwargs = {}
    engine = Omni(
        model=model,
        parallel_config=DiffusionParallelConfig(ulysses_degree=1),
        trust_remote_code=True,
        enforce_eager=True,
        **offload_kwargs,
    )

    def _one():
        t0 = time.perf_counter()
        outputs = engine.generate(
            "A quiet cinematic night scene with matching ambient sound.",
            OmniDiffusionSamplingParams(
                height=256,
                width=448,
                num_frames=29,
                fps=24,
                num_inference_steps=args.steps,
                seed=42,
                output_type="np",
                extra_args={"task": "t2va", "duration": 4.0, "aspect_ratio": "16:9", "flow_shift": 12.0, "audio_flow_shift": 3.0},
            ),
            use_tqdm=False,
        )
        dt = time.perf_counter() - t0
        return outputs[0], dt

    try:
        for _ in range(args.warmup):
            _one()
        times = []
        last = None
        for _ in range(args.runs):
            last, dt = _one()
            times.append(dt)
    finally:
        engine.close()

    video = np.asarray(last.images[0])
    mm = last.multimodal_output
    audio = np.asarray(mm["audio"]) if mm is not None else None

    result = {
        "offload": args.offload,
        "steps": args.steps,
        "runs": args.runs,
        "times_s": times,
        "time_median_s": statistics.median(times),
        "time_min_s": min(times),
        "video_shape": list(video.shape),
        "video_sha256": _sha(video),
    }
    if audio is not None:
        result["audio_sha256"] = _sha(audio)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
