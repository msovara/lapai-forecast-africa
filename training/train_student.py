"""Train LapAI student with staged losses A / B / C."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from evaluation.masks import africa_hw_mask
from lapai_inference.dataset import LapAIZarrDataset, collate_lapai_batch, open_cache_readonly
from lapai_inference.model import LapAIStudentCNN, LapAIStudentConfig
from lapai_inference.cache_schema import cosine_latitude_weights, lat_lon_mesh
from utils.losses_distillation import (
    FeatureDistillationHead,
    apply_soft_physical_constraints,
    combined_distillation_loss,
    normalize_channels,
)


# Fallback Cout=3 (tp, msl, 2t) if cache stats unavailable (synth smoke).
_FALLBACK_MEAN = torch.tensor([5e-4, 1.01e5, 278.0], dtype=torch.float32)
_FALLBACK_STD = torch.tensor([2e-3, 1.3e3, 20.0], dtype=torch.float32)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def synth_batch(cfg: LapAIStudentConfig, batch: int, device: torch.device) -> dict:
    x = torch.randn(batch, cfg.in_channels_raw, cfg.lat, cfg.lon, device=device)
    era5 = torch.randn(batch, cfg.out_channels, cfg.lat, cfg.lon, device=device)
    teacher_pred = torch.randn(batch, cfg.out_channels, cfg.lat, cfg.lon, device=device)
    d10 = 128
    d14 = 128
    feat10 = torch.randn(batch, d10, cfg.lat, cfg.lon, device=device)
    feat14 = torch.randn(batch, d14, cfg.lat, cfg.lon, device=device)
    return {"x": x, "era5": era5, "teacher_pred": teacher_pred, "feat10": feat10, "feat14": feat14}


def make_cos_lat_weights(cfg: LapAIStudentConfig, device: torch.device) -> torch.Tensor:
    lat, _ = lat_lon_mesh(cfg.lat, cfg.lon)
    w = cosine_latitude_weights(lat)
    return torch.as_tensor(w, dtype=torch.float32, device=device).squeeze()


def compute_target_stats_from_cache(
    cache: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std of era5_target over the full cache (compute once)."""
    g = open_cache_readonly(cache)
    era5 = np.asarray(g["era5_target"][:], dtype=np.float64)  # (T,C,H,W)
    # Reduce over T,H,W → (C,)
    mean = era5.mean(axis=(0, 2, 3))
    std = era5.std(axis=(0, 2, 3))
    std = np.maximum(std, 1e-6)
    print(
        "[Track B] target stats from cache "
        + " ".join(f"c{i}:mean={mean[i]:.4g}/std={std[i]:.4g}" for i in range(len(mean)))
    )
    return (
        torch.as_tensor(mean, dtype=torch.float32, device=device),
        torch.as_tensor(std, dtype=torch.float32, device=device),
    )


def compute_input_stats_from_cache(
    cache: Path, device: torch.device, max_samples: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std of state_in (subsample T for speed)."""
    g = open_cache_readonly(cache)
    t_all = int(g["state_in"].shape[0])
    n = min(max_samples, t_all)
    idx = np.linspace(0, t_all - 1, n, dtype=np.int64)
    x = np.asarray(g["state_in"][idx.tolist()], dtype=np.float64)  # (n,C,H,W)
    mean = x.mean(axis=(0, 2, 3))
    std = np.maximum(x.std(axis=(0, 2, 3)), 1e-6)
    print(
        "[Track B] input stats from cache "
        + " ".join(f"c{i}:mean={mean[i]:.4g}/std={std[i]:.4g}" for i in range(min(5, len(mean))))
        + (" ..." if len(mean) > 5 else "")
    )
    return (
        torch.as_tensor(mean, dtype=torch.float32, device=device),
        torch.as_tensor(std, dtype=torch.float32, device=device),
    )


def batch_generator_loader(
    loader: DataLoader, device: torch.device, steps: int
) -> Iterator[dict]:
    it = iter(loader)
    for _ in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        yield {k: v.to(device) for k, v in b.items()}


def batch_generator_synth(
    cfg: LapAIStudentConfig, device: torch.device, batch_size: int, steps: int
) -> Iterator[dict]:
    for _ in range(steps):
        yield synth_batch(cfg, batch_size, device)


def staged_weight(
    epoch: int,
    start_epoch: int,
    target: float,
    ramp_epochs: int,
) -> float:
    """0 before start; linear ramp to target over ramp_epochs (inclusive of start)."""
    if epoch < start_epoch or target <= 0:
        return 0.0
    if ramp_epochs <= 1:
        return float(target)
    t = min(1.0, (epoch - start_epoch + 1) / float(ramp_epochs))
    return float(target) * t


def train_epoch(
    net: LapAIStudentCNN,
    opt: optim.Optimizer,
    heads: tuple[FeatureDistillationHead, FeatureDistillationHead],
    batches: Iterator[dict],
    w_lat: torch.Tensor,
    *,
    beta_w: float,
    gamma_w: float,
    spectral_k_max: int | None,
    steps: int,
    alpha: float = 1.0,
    target_mean: Optional[torch.Tensor] = None,
    target_std: Optional[torch.Tensor] = None,
    input_mean: Optional[torch.Tensor] = None,
    input_std: Optional[torch.Tensor] = None,
    grad_clip: float = 1.0,
    physical_constraints: bool = True,
    tp_mode: str = "relu",
    channel_weights: Optional[torch.Tensor] = None,
    africa_mix: float = 0.0,
    africa_spatial: Optional[torch.Tensor] = None,
    precip_log1p_weight: float = 0.0,
    precip_wet_boost: float = 4.0,
    precip_underpred_weight: float = 0.0,
    precip_pod_weight: float = 0.0,
    precip_dry_collapse_weight: float = 0.0,
) -> tuple[float, dict[str, float]]:
    net.train()
    running = 0.0
    part_sums = {"L_A": 0.0, "L_B": 0.0, "L_C": 0.0, "L_tp": 0.0}
    gen = batches
    params = list(net.parameters()) + list(heads[0].parameters()) + list(heads[1].parameters())
    for _ in range(steps):
        b = next(gen)
        opt.zero_grad(set_to_none=True)
        x = b["x"]
        if input_mean is not None and input_std is not None:
            x = normalize_channels(x, input_mean, input_std)
        out = net(x)
        pred = out["pred"]
        if physical_constraints:
            pred = apply_soft_physical_constraints(pred, tp_mode=tp_mode)
        use_b = beta_w > 0.0
        use_c = gamma_w > 0.0
        loss, parts = combined_distillation_loss(
            pred,
            b["era5"],
            b["teacher_pred"],
            out["feat_stage2"],
            out["feat_stage3"],
            b["feat10"],
            b["feat14"],
            w_lat,
            heads,
            alpha=alpha,
            beta=beta_w,
            gamma=gamma_w,
            use_feature_loss=use_b,
            use_spectral_loss=use_c,
            spectral_k_max=spectral_k_max,
            target_mean=target_mean,
            target_std=target_std,
            channel_weights=channel_weights,
            africa_mix=africa_mix,
            africa_spatial_weights=africa_spatial,
            precip_log1p_weight=precip_log1p_weight,
            precip_wet_boost=precip_wet_boost,
            precip_underpred_weight=precip_underpred_weight,
            precip_pod_weight=precip_pod_weight,
            precip_dry_collapse_weight=precip_dry_collapse_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss L_A={float(parts['L_A'])} "
                f"L_B={float(parts['L_B'])} L_C={float(parts['L_C'])} "
                f"L_tp={float(parts['L_tp'])}"
            )
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        running += float(loss.detach())
        for k in part_sums:
            part_sums[k] += float(parts[k])
    avg_parts = {k: v / steps for k, v in part_sums.items()}
    return running / steps, avg_parts


def cfg_from_dataset(ds: LapAIZarrDataset, yml: dict[str, Any] | None = None) -> LapAIStudentConfig:
    """Default demo alignment: Cin = 65 = 13 * 5 multilevel stack."""
    yml = yml or {}
    kwargs: dict[str, Any] = {
        "lat": ds.lat,
        "lon": ds.lon,
        "in_channels_raw": ds.cin,
        "out_channels": ds.cout,
    }
    if "base_channels" in yml:
        kwargs["base_channels"] = int(yml["base_channels"])
    if "stages" in yml:
        kwargs["stages"] = tuple(yml["stages"])
    return LapAIStudentConfig(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="Track B student training")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/student_global.yaml"),
        help="Student geometry + teacher_ckpt handoff (K1)",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cache", type=Path, default=None, help="Path to Zarr store with lapai_cache/")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--steps_per_epoch", type=int, default=32)
    p.add_argument(
        "--teacher-ckpt",
        type=Path,
        default=None,
        help="Override teacher_ckpt from yaml (Track A K1: models/teacher_pruned.ckpt)",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume student (+ heads) weights from a prior ckpt (e.g. stable_v2)",
    )
    args = p.parse_args()

    yml = _read_yaml(args.config) if args.config else {}
    distill_path = Path(yml.get("distill_config", "configs/student_distill.yaml"))
    distill = _read_yaml(distill_path)

    teacher_ckpt = Path(
        args.teacher_ckpt
        or yml.get("teacher_ckpt")
        or distill.get("teacher_ckpt")
        or "models/teacher_pruned.ckpt"
    )
    print(f"[Track B] config={args.config} teacher_ckpt={teacher_ckpt} (exists={teacher_ckpt.exists()})")

    epochs = int(args.epochs if args.epochs is not None else yml.get("epochs", 25))
    out = args.out or Path(yml.get("output_ckpt", "models/student_global.ckpt"))
    cache = args.cache or (Path(yml["cache"]) if yml.get("cache") else None)
    resume = args.resume or (Path(yml["resume"]) if yml.get("resume") else None)

    epoch_feat = int(yml.get("epoch_start_feature_loss", 18))
    epoch_spec = int(yml.get("epoch_start_spectral_loss", 22))
    alpha = float(distill.get("alpha", 1.0))
    beta_target = float(distill.get("beta", 0.05))
    gamma_target = float(distill.get("gamma", 0.05))
    beta_ramp = int(distill.get("beta_ramp_epochs", yml.get("beta_ramp_epochs", 5)))
    gamma_ramp = int(distill.get("gamma_ramp_epochs", yml.get("gamma_ramp_epochs", 4)))
    spectral_k_warmup = distill.get("spectral_k_warmup", 40)
    spectral_k_warmup = int(spectral_k_warmup) if spectral_k_warmup is not None else None
    grad_clip = float(yml.get("grad_clip", distill.get("grad_clip", 1.0)))
    normalize_targets = bool(yml.get("normalize_targets", True))
    normalize_inputs = bool(yml.get("normalize_inputs", False))
    physical_constraints = bool(yml.get("physical_constraints", True))
    tp_mode = str(yml.get("tp_mode", distill.get("tp_mode", "relu")))
    africa_mix = float(yml.get("africa_mix", distill.get("africa_mix", 0.0)))
    precip_log1p_weight = float(
        yml.get("precip_log1p_weight", distill.get("precip_log1p_weight", 0.0))
    )
    precip_wet_boost = float(yml.get("precip_wet_boost", distill.get("precip_wet_boost", 4.0)))
    precip_underpred_weight = float(
        yml.get("precip_underpred_weight", distill.get("precip_underpred_weight", 0.0))
    )
    precip_pod_weight = float(yml.get("precip_pod_weight", distill.get("precip_pod_weight", 0.0)))
    precip_dry_collapse_weight = float(
        yml.get("precip_dry_collapse_weight", distill.get("precip_dry_collapse_weight", 0.0))
    )
    tp_logit_boost = float(yml.get("tp_logit_boost", distill.get("tp_logit_boost", 0.0)))
    # Cout order: tp, msl, 2t — up-weight tp/2t when msl already near gate.
    cw_raw = yml.get("channel_weights", distill.get("channel_weights"))
    channel_weights_list = [float(x) for x in cw_raw] if cw_raw is not None else None

    device = torch.device(args.device)

    input_mean = input_std = None
    if cache is not None:
        ds = LapAIZarrDataset(cache)
        cfg = cfg_from_dataset(ds, yml)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_lapai_batch,
            pin_memory=device.type == "cuda",
        )
        batch_factory: Callable[[], Iterator[dict]] = lambda: batch_generator_loader(
            loader, device, args.steps_per_epoch
        )
        heads = (
            FeatureDistillationHead(cfg.base_channels, ds.d_feat10, proj_dim=128),
            FeatureDistillationHead(cfg.base_channels, ds.d_feat14, proj_dim=128),
        )
        if normalize_targets:
            target_mean, target_std = compute_target_stats_from_cache(cache, device)
        else:
            target_mean = target_std = None
        if normalize_inputs:
            input_mean, input_std = compute_input_stats_from_cache(cache, device)
    else:
        model_kwargs: dict[str, Any] = {}
        if "lat" in yml:
            model_kwargs["lat"] = int(yml["lat"])
        if "lon" in yml:
            model_kwargs["lon"] = int(yml["lon"])
        if "base_channels" in yml:
            model_kwargs["base_channels"] = int(yml["base_channels"])
        if "stages" in yml:
            model_kwargs["stages"] = tuple(yml["stages"])
        cfg = LapAIStudentConfig(**model_kwargs)
        batch_factory = lambda: batch_generator_synth(
            cfg, device, args.batch_size, args.steps_per_epoch
        )
        heads = (
            FeatureDistillationHead(cfg.base_channels, 128, proj_dim=128),
            FeatureDistillationHead(cfg.base_channels, 128, proj_dim=128),
        )
        if normalize_targets:
            target_mean = _FALLBACK_MEAN.to(device)
            target_std = _FALLBACK_STD.to(device)
            if cfg.out_channels != 3:
                target_mean = torch.zeros(cfg.out_channels, device=device)
                target_std = torch.ones(cfg.out_channels, device=device)
        else:
            target_mean = target_std = None

    net = LapAIStudentCNN(cfg).to(device)
    for h in heads:
        h.to(device)

    if resume is not None:
        if not resume.exists():
            raise SystemExit(f"--resume not found: {resume}")
        blob_in = torch.load(resume, map_location="cpu", weights_only=False)
        net.load_state_dict(blob_in["model"])
        hs = blob_in.get("heads") or []
        for i, h in enumerate(heads):
            if i < len(hs):
                h.load_state_dict(hs[i])
        print(f"[Track B] resumed model+heads from {resume}")
        if tp_logit_boost != 0.0 and net.head.bias is not None and net.head.bias.numel() >= 1:
            with torch.no_grad():
                net.head.bias[0] = net.head.bias[0] + float(tp_logit_boost)
            print(f"[Track B] tp_logit_boost={tp_logit_boost} applied to head bias[0]")
    elif (
        target_mean is not None
        and net.head.bias is not None
        and target_mean.numel() == cfg.out_channels
    ):
        # Bias head toward physical channel means so early grads aren't stuck under dead clamps.
        with torch.no_grad():
            net.head.bias.copy_(target_mean.to(device=net.head.bias.device, dtype=net.head.bias.dtype))
            nn.init.zeros_(net.head.weight)

    channel_weights = None
    if channel_weights_list is not None:
        if len(channel_weights_list) != cfg.out_channels:
            raise SystemExit(
                f"channel_weights len={len(channel_weights_list)} != out_channels={cfg.out_channels}"
            )
        channel_weights = torch.tensor(channel_weights_list, dtype=torch.float32, device=device)
        print(f"[Track B] channel_weights (tp,msl,2t)={channel_weights_list}")

    africa_spatial = None
    if africa_mix > 0.0:
        lat_np, lon_np = lat_lon_mesh(cfg.lat, cfg.lon)
        africa_spatial = torch.as_tensor(
            africa_hw_mask(lat_np, lon_np), dtype=torch.float32, device=device
        )
        print(
            f"[Track B] africa_mix={africa_mix} mask_frac={float(africa_spatial.mean()):.4f} "
            f"tp_mode={tp_mode} precip_log1p_weight={precip_log1p_weight} "
            f"wet_boost={precip_wet_boost} underpred={precip_underpred_weight} "
            f"pod={precip_pod_weight} dry_collapse={precip_dry_collapse_weight} "
            f"normalize_inputs={normalize_inputs}"
        )
    else:
        print(
            f"[Track B] africa_mix=0 tp_mode={tp_mode} precip_log1p_weight={precip_log1p_weight} "
            f"wet_boost={precip_wet_boost} underpred={precip_underpred_weight} "
            f"pod={precip_pod_weight} dry_collapse={precip_dry_collapse_weight} "
            f"normalize_inputs={normalize_inputs}"
        )

    opt = optim.AdamW(
        list(net.parameters()) + list(heads[0].parameters()) + list(heads[1].parameters()),
        lr=3e-4,
    )

    w_lat = make_cos_lat_weights(cfg, device)

    out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[Track B] schedule feat@{epoch_feat} (β→{beta_target}, ramp={beta_ramp}) "
        f"spec@{epoch_spec} (γ→{gamma_target}, ramp={gamma_ramp}) "
        f"grad_clip={grad_clip} norm={normalize_targets} phys={physical_constraints} "
        f"resume={resume}"
    )
    for epoch in range(1, epochs + 1):
        beta_w = staged_weight(epoch, epoch_feat, beta_target, beta_ramp)
        gamma_w = staged_weight(epoch, epoch_spec, gamma_target, gamma_ramp)
        kspec = spectral_k_warmup if epoch <= 20 else None
        batches = batch_factory()
        loss_m, parts = train_epoch(
            net,
            opt,
            heads,
            batches,
            w_lat,
            beta_w=beta_w,
            gamma_w=gamma_w,
            spectral_k_max=kspec,
            steps=args.steps_per_epoch,
            alpha=alpha,
            target_mean=target_mean,
            target_std=target_std,
            input_mean=input_mean,
            input_std=input_std,
            grad_clip=grad_clip,
            physical_constraints=physical_constraints,
            tp_mode=tp_mode,
            channel_weights=channel_weights,
            africa_mix=africa_mix,
            africa_spatial=africa_spatial,
            precip_log1p_weight=precip_log1p_weight,
            precip_wet_boost=precip_wet_boost,
            precip_underpred_weight=precip_underpred_weight,
            precip_pod_weight=precip_pod_weight,
            precip_dry_collapse_weight=precip_dry_collapse_weight,
        )
        print(
            f"epoch {epoch} loss={loss_m:.6f} "
            f"L_A={parts['L_A']:.6f} L_B={parts['L_B']:.6f} L_C={parts['L_C']:.6f} "
            f"L_tp={parts['L_tp']:.6f} "
            f"beta_w={beta_w:.4f} gamma_w={gamma_w:.4f} "
            f"cache={cache} teacher={teacher_ckpt}"
        )

    blob: dict[str, Any] = {
        "model": net.state_dict(),
        "heads": [h.state_dict() for h in heads],
        "cfg": asdict(cfg),
        "teacher_ckpt": str(teacher_ckpt),
        "normalize_targets": normalize_targets,
        "normalize_inputs": normalize_inputs,
        "physical_constraints": physical_constraints,
        "tp_mode": tp_mode,
        "africa_mix": africa_mix,
        "precip_log1p_weight": precip_log1p_weight,
        "precip_wet_boost": precip_wet_boost,
        "precip_underpred_weight": precip_underpred_weight,
        "precip_pod_weight": precip_pod_weight,
        "precip_dry_collapse_weight": precip_dry_collapse_weight,
        "tp_logit_boost": tp_logit_boost,
        "channel_weights": channel_weights_list,
        "resume_from": str(resume) if resume is not None else None,
    }
    if target_mean is not None and target_std is not None:
        blob["target_mean"] = target_mean.detach().cpu()
        blob["target_std"] = target_std.detach().cpu()
    if input_mean is not None and input_std is not None:
        blob["input_mean"] = input_mean.detach().cpu()
        blob["input_std"] = input_std.detach().cpu()
    torch.save(blob, out)
    print("saved", out.resolve(), "params", net.count_parameters())


if __name__ == "__main__":
    main()
