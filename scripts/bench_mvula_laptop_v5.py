#!/usr/bin/env python3
"""CPU-only laptop-style inference benchmark for frozen v5 student.

Runs with CUDA_VISIBLE_DEVICES unset/empty. Writes:
  reports/MVULA_LAPTOP_BENCHMARK.json

Caveat: when run on Cassava CPU this is a proxy demo, not a consumer laptop.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Force CPU before torch import side-effects.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_REPO = Path(os.environ.get("LAPAI_REPO", "")).resolve() if os.environ.get("LAPAI_REPO") else Path(__file__).resolve().parents[1]
if not (_REPO / "models").is_dir():
    # When piped to /tmp/... use cwd or Cassava default.
    for cand in (Path.cwd(), Path("/local/Mthetho/lapai-forecast")):
        if (cand / "models").is_dir():
            _REPO = cand.resolve()
            break
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

import torch  # noqa: E402


def _fsize(p: Path | None) -> int | None:
    if p is None or not p.exists():
        return None
    return p.stat().st_size


def main() -> int:
    from evaluation.trackB_gate import _load_student
    from utils.losses_distillation import apply_soft_physical_constraints, normalize_channels

    try:
        import psutil

        proc = psutil.Process()
        rss0 = proc.memory_info().rss
    except Exception:  # noqa: BLE001
        proc = None
        rss0 = 0

    ckpt = _REPO / "models/student_global_stable_v5.ckpt"
    teacher = _REPO / "models/teacher_pruned.ckpt"
    teacher_real = teacher.resolve() if teacher.exists() else None
    if not ckpt.is_file():
        raise SystemExit(f"missing ckpt: {ckpt}")

    device = torch.device("cpu")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4") or 4))

    t0 = time.perf_counter()
    net, meta = _load_student(ckpt, device)
    load_s = time.perf_counter() - t0
    rss_after_load = proc.memory_info().rss if proc else 0

    n_params = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)

    x = torch.randn(1, 65, 181, 360, dtype=torch.float32)
    if meta.get("normalize_inputs") and meta.get("input_mean") is not None:
        im = torch.as_tensor(meta["input_mean"], dtype=torch.float32)
        istd = torch.as_tensor(meta["input_std"], dtype=torch.float32)
        x = normalize_channels(x, im, istd)

    with torch.no_grad():
        out = net(x)["pred"]
        if bool(meta.get("physical_constraints", True)):
            out = apply_soft_physical_constraints(
                out, tp_mode=str(meta.get("tp_mode", "relu"))
            )

    times: list[float] = []
    rss_peak = rss_after_load
    with torch.no_grad():
        for _ in range(8):
            t1 = time.perf_counter()
            out = net(x)["pred"]
            if bool(meta.get("physical_constraints", True)):
                out = apply_soft_physical_constraints(
                    out, tp_mode=str(meta.get("tp_mode", "relu"))
                )
            _ = out.numpy()
            times.append(time.perf_counter() - t1)
            if proc is not None:
                rss_peak = max(rss_peak, proc.memory_info().rss)

    timed = times[1:]
    mean_s = sum(timed) / len(timed)
    stud_b = _fsize(ckpt)
    teach_b = _fsize(teacher_real)

    host = os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "unknown"
    result = {
        "host": host,
        "host_class": "cassava_cpu_proxy"
        if "cassava" in host.lower() or host.startswith("gpu")
        else "local",
        "caveat": (
            "CPU-only timed demo (CUDA_VISIBLE_DEVICES empty). "
            "On Cassava this is a proxy for laptop inference, not a consumer i7/16GB machine. "
            "Peak RSS is process memory on this host."
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
        "mkl_num_threads": int(os.environ.get("MKL_NUM_THREADS", "0") or 0),
        "cpu_count_logical": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "student_ckpt": str(ckpt),
        "student_ckpt_bytes": stud_b,
        "student_ckpt_mib": round(stud_b / 1024 / 1024, 3) if stud_b else None,
        "teacher_ckpt": str(teacher_real) if teacher_real else None,
        "teacher_ckpt_bytes": teach_b,
        "teacher_ckpt_mib": round(teach_b / 1024 / 1024, 3) if teach_b else None,
        "size_reduction_factor_disk": round(teach_b / stud_b, 2)
        if stud_b and teach_b
        else None,
        "student_params": int(n_params),
        "student_trainable_params": int(n_train),
        "student_params_m": round(n_params / 1e6, 3),
        "out_channels": int(getattr(net.cfg, "out_channels", -1)),
        "in_channels_raw": int(getattr(net.cfg, "in_channels_raw", -1)),
        "load_seconds": round(load_s, 4),
        "rss_before_bytes": rss0,
        "rss_after_load_bytes": rss_after_load,
        "rss_peak_bytes": rss_peak,
        "rss_peak_mib": round(rss_peak / 1024 / 1024, 1) if rss_peak else None,
        "rss_delta_load_mib": round((rss_after_load - rss0) / 1024 / 1024, 1)
        if rss_after_load
        else None,
        "step_seconds": [round(t, 4) for t in timed],
        "mean_step_seconds": round(mean_s, 4),
        "af_package_4leads_infer_only_seconds": round(mean_s * 4, 4),
        "hypothetical_40step_head_only_seconds": round(mean_s * 40, 4),
        "free_run_supported": False,
        "gpu_required": False,
        "protocol_note": (
            "v5 Cout=3 is analysis-forced only. Each AF lead needs its own ERA5 IC "
            "(IC fetch/build not timed here). Free-run 10-day rollout is NOT supported."
        ),
        "pred_shape": list(out.shape),
        "can_run_on_normal_laptop_estimate": (
            "Likely yes for torch CPU inference of the ~9 MiB student head "
            "(params ~2M; step time typically seconds on a mid-range CPU). "
            "Full AF package also needs ERA5 IC access (ARCO/CDS) and Python deps; "
            "ONNX packaging / 16 GB end-to-end demo still TO COMPLETE."
        ),
    }

    out_json = _REPO / "reports/MVULA_LAPTOP_BENCHMARK.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"WROTE {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
