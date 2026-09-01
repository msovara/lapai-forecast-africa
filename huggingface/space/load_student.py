"""Minimal Mvula v5 student load + climate-IC forward for Hugging Face Spaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

# Prefer installed package when Space clones / installs the GitHub repo.
try:
    from lapai_inference.model import LapAIStudentCNN, LapAIStudentConfig
except ImportError:  # pragma: no cover - Space may vendor path later
    LapAIStudentCNN = None  # type: ignore[misc, assignment]
    LapAIStudentConfig = None  # type: ignore[misc, assignment]


def load_student(ckpt_path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    if LapAIStudentCNN is None or LapAIStudentConfig is None:
        raise ImportError(
            "lapai_inference is not installed. "
            "Add the GitHub package to requirements or vendor lapai_inference/model.py."
        )
    path = Path(ckpt_path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    allowed = {f.name for f in LapAIStudentConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    cfg = LapAIStudentConfig(**{k: v for k, v in dict(blob.get("cfg") or {}).items() if k in allowed})
    net = LapAIStudentCNN(cfg)
    net.load_state_dict(blob["model"])
    net.to(torch.device(device))
    net.eval()
    meta = {
        "normalize_inputs": bool(blob.get("normalize_inputs", False)),
        "input_mean": blob.get("input_mean"),
        "input_std": blob.get("input_std"),
        "cfg": cfg,
    }
    return net, meta


def climate_forward_t2m_celsius(net: torch.nn.Module, meta: dict[str, Any], seed: int = 0) -> np.ndarray:
    """One forward from climate(+noise) IC → global 2t field in °C (H, W)."""
    cfg = meta["cfg"]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(1, cfg.in_channels_raw, cfg.lat, cfg.lon, generator=g, dtype=torch.float32)
    if meta.get("input_mean") is not None and meta.get("input_std") is not None:
        im = torch.as_tensor(meta["input_mean"], dtype=torch.float32).view(1, -1, 1, 1)
        istd = torch.as_tensor(meta["input_std"], dtype=torch.float32).view(1, -1, 1, 1)
        x = im + 0.05 * istd * x
        if meta.get("normalize_inputs"):
            x = (x - im) / istd.clamp_min(1e-6)
    device = next(net.parameters()).device
    with torch.no_grad():
        pred = net(x.to(device))["pred"][0, 2].detach().cpu().numpy()  # 2t
    # Heuristic: if values look like Kelvin, convert
    if float(np.nanmean(pred)) > 100:
        pred = pred - 273.15
    return pred.astype(np.float32)


def africa_crop(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crude 1° crop to Africa box lon −20…55, lat −40…40 (0…360 mesh assumed)."""
    lat = np.linspace(-90.0, 90.0, field.shape[0])
    lon = np.linspace(0.0, 360.0 - 360.0 / field.shape[1], field.shape[1])
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon180)
    lon180 = lon180[order]
    field = field[:, order]
    lat_ok = (lat >= -40) & (lat <= 40)
    lon_ok = (lon180 >= -20) & (lon180 <= 55)
    return field[np.ix_(lat_ok, lon_ok)], lat[lat_ok], lon180[lon_ok]
