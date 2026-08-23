"""Soft attention-head pruning for Anemoi GraphTransformer checkpoints.

A1 GNN coarsened teachers have no heads — refuse those checkpoints.
Operates on Lightning / anemoi inference state_dicts by zeroing head slices in
``lin_query|key|value|self|edge`` and the matching ``projection`` input columns.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

_LIN_RE = re.compile(
    r"(?P<prefix>.*\.)(?P<lin>lin_(?:query|key|value|self|edge))\.weight$"
)
_PROJ_RE = re.compile(r"(?P<prefix>.*\.)projection\.weight$")


@dataclass
class HeadScore:
    block_prefix: str  # e.g. model.processor.proc.0.
    head: int
    score: float


def _unwrap_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        # anemoi inference package layout
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
        # raw state dict
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    # anemoi inference-last.ckpt loads as AnemoiModelInterface (nn.Module)
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        sd = obj.state_dict()
        if isinstance(sd, dict) and any(isinstance(v, torch.Tensor) for v in sd.values()):
            return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    raise TypeError(f"unrecognised checkpoint object type={type(obj)}")


def _save_pruned_ckpt(blob: Any, state: dict[str, torch.Tensor], out_ckpt: Path) -> None:
    """Write masked weights, preserving Anemoi inference / Lightning layouts."""
    out_ckpt = Path(out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(blob, dict) and "state_dict" in blob:
        blob["state_dict"] = state
        torch.save(blob, out_ckpt)
        return
    if isinstance(blob, dict) and "model_state_dict" in blob:
        blob["model_state_dict"] = state
        torch.save(blob, out_ckpt)
        return
    if hasattr(blob, "load_state_dict") and callable(blob.load_state_dict):
        # AnemoiModelInterface: do not re-pickle the full module (legacy Triton
        # symbols / stubs). Write Lightning-style weights for load_weights_only.
        torch.save({"state_dict": state}, out_ckpt)
        return
    torch.save({"state_dict": state}, out_ckpt)


class _GraphTransformerFunctionStub(torch.autograd.Function):
    """Module-level stub so legacy anemoi inference pickles can unpickle."""

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG004
        raise RuntimeError("GraphTransformerFunction stub — use state_dict only")

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:  # noqa: ARG004
        raise RuntimeError("GraphTransformerFunction stub — use state_dict only")


def _stub_legacy_anemoi_pickle_symbols() -> None:
    """Allow loading inference ckpts pickled under older anemoi-models.

    Some Lengau checkpoints reference ``anemoi.models.triton.gt.GraphTransformerFunction``,
    which newer anemoi-models dropped. We only need weights for soft-mask prune.
    """
    try:
        import anemoi.models.triton.gt as gt  # noqa: PLC0415
    except Exception:
        return
    if not hasattr(gt, "GraphTransformerFunction"):
        gt.GraphTransformerFunction = _GraphTransformerFunctionStub  # type: ignore[attr-defined]


def load_ckpt_blob(path: Path) -> tuple[Any, dict[str, torch.Tensor]]:
    _stub_legacy_anemoi_pickle_symbols()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob, _unwrap_state_dict(blob)


def infer_num_heads_from_metadata(ckpt_path: Path) -> int | None:
    """Best-effort read of processor.num_heads from anemoi metadata sidecar/zip."""
    import zipfile

    p = Path(ckpt_path)
    candidates = [
        p.parent / "inference-last" / "anemoi-metadata" / "anemoi.json",
        p.with_name("anemoi.json"),
    ]
    for c in candidates:
        if c.is_file():
            meta = json.loads(c.read_text(encoding="utf-8"))
            try:
                return int(meta["config"]["model"]["processor"]["num_heads"])
            except (KeyError, TypeError, ValueError):
                pass
    # zip-style inference ckpt
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as zf:
            for name in zf.namelist():
                if name.endswith("anemoi-metadata/anemoi.json") or name.endswith("anemoi.json"):
                    meta = json.loads(zf.read(name))
                    try:
                        return int(meta["config"]["model"]["processor"]["num_heads"])
                    except (KeyError, TypeError, ValueError):
                        continue
    return None


def discover_gt_lin_keys(
    state: dict[str, torch.Tensor],
    *,
    scope: str = "processor",
) -> dict[str, list[str]]:
    """Map block prefix -> list of lin_*.weight keys (and later projection)."""
    blocks: dict[str, list[str]] = {}
    for key in state:
        m = _LIN_RE.match(key)
        if not m:
            continue
        prefix = m.group("prefix")
        if scope == "processor" and "processor" not in prefix:
            continue
        if scope == "encoder" and "encoder" not in prefix:
            continue
        if scope == "decoder" and "decoder" not in prefix:
            continue
        if scope == "all":
            pass
        elif scope not in ("processor", "encoder", "decoder"):
            raise ValueError(f"unknown scope {scope}")
        blocks.setdefault(prefix, []).append(key)
    return blocks


def assert_has_attention_heads(state: dict[str, torch.Tensor], scope: str = "processor") -> None:
    blocks = discover_gt_lin_keys(state, scope=scope)
    if not blocks:
        raise RuntimeError(
            "No GraphTransformer lin_query/key/value weights found in checkpoint. "
            "A1 GNN coarsened teachers cannot be head-pruned. "
            "Fine-tune GraphTransformer on O96 first (configs/trackA_gt_coarsen.yaml)."
        )


def score_heads_weight_l1(
    state: dict[str, torch.Tensor],
    *,
    num_heads: int,
    scope: str = "processor",
) -> list[HeadScore]:
    """Importance = mean |W| over each head's output slice across lin_* mats."""
    assert_has_attention_heads(state, scope=scope)
    blocks = discover_gt_lin_keys(state, scope=scope)
    scores: list[HeadScore] = []
    for prefix, keys in blocks.items():
        head_acc = [0.0] * num_heads
        head_n = [0] * num_heads
        for key in keys:
            w = state[key]
            if w.ndim != 2:
                continue
            out_f = w.shape[0]
            if out_f % num_heads != 0:
                continue
            d = out_f // num_heads
            for h in range(num_heads):
                sl = w[h * d : (h + 1) * d]
                head_acc[h] += float(sl.abs().mean())
                head_n[h] += 1
        for h in range(num_heads):
            if head_n[h] == 0:
                continue
            scores.append(HeadScore(prefix, h, head_acc[h] / head_n[h]))
    return scores


def select_heads_to_drop(
    scores: list[HeadScore],
    *,
    fraction: float,
    num_heads: int,
) -> dict[str, list[int]]:
    """Per-block: drop the lowest-scoring ceil(fraction * num_heads) heads (min 1 if fraction>0)."""
    n_drop = max(1, int(math.ceil(fraction * num_heads))) if fraction > 0 else 0
    by_block: dict[str, list[HeadScore]] = {}
    for s in scores:
        by_block.setdefault(s.block_prefix, []).append(s)
    drop: dict[str, list[int]] = {}
    for prefix, lst in by_block.items():
        ordered = sorted(lst, key=lambda x: x.score)
        drop[prefix] = [s.head for s in ordered[:n_drop]]
    return drop


def apply_soft_head_mask(
    state: dict[str, torch.Tensor],
    drop: dict[str, list[int]],
    *,
    num_heads: int,
) -> dict[str, list[int]]:
    """In-place zero head slices; returns applied drop map."""
    for prefix, heads in drop.items():
        if not heads:
            continue
        for lin in ("lin_query", "lin_key", "lin_value", "lin_self", "lin_edge"):
            key = f"{prefix}{lin}.weight"
            if key not in state:
                continue
            w = state[key]
            d = w.shape[0] // num_heads
            for h in heads:
                w.data[h * d : (h + 1) * d].zero_()
            bkey = f"{prefix}{lin}.bias"
            if bkey in state and state[bkey] is not None:
                b = state[bkey]
                if b.numel() == num_heads * d:
                    for h in heads:
                        b.data[h * d : (h + 1) * d].zero_()
        proj = f"{prefix}projection.weight"
        if proj in state:
            w = state[proj]  # [out, attn_channels]
            d = w.shape[1] // num_heads
            for h in heads:
                w.data[:, h * d : (h + 1) * d].zero_()
    return drop


def prune_checkpoint(
    in_ckpt: Path,
    out_ckpt: Path,
    *,
    fraction: float = 0.10,
    num_heads: int | None = None,
    scope: str = "processor",
    importance: str = "weight_l1",
) -> dict[str, Any]:
    """Load ckpt, soft-mask lowest heads, write out_ckpt; return report dict."""
    blob, state = load_ckpt_blob(in_ckpt)
    nh = num_heads or infer_num_heads_from_metadata(in_ckpt)
    if nh is None:
        # infer from first lin_query out dim vs common head counts
        blocks = discover_gt_lin_keys(state, scope=scope)
        if not blocks:
            assert_has_attention_heads(state, scope=scope)
        sample_key = next(iter(next(iter(blocks.values()))))
        out_f = state[sample_key].shape[0]
        for cand in (16, 8, 32, 4):
            if out_f % cand == 0:
                nh = cand
                break
        if nh is None:
            raise RuntimeError(f"cannot infer num_heads from {sample_key} out={out_f}")

    if importance != "weight_l1":
        raise NotImplementedError(f"importance={importance} not implemented yet; use weight_l1")

    scores = score_heads_weight_l1(state, num_heads=nh, scope=scope)
    drop = select_heads_to_drop(scores, fraction=fraction, num_heads=nh)
    apply_soft_head_mask(state, drop, num_heads=nh)

    _save_pruned_ckpt(blob, state, Path(out_ckpt))

    report = {
        "input_ckpt": str(in_ckpt),
        "output_ckpt": str(out_ckpt),
        "num_heads": nh,
        "fraction": fraction,
        "scope": scope,
        "importance": importance,
        "dropped_heads": drop,
        "scores": [asdict(s) for s in sorted(scores, key=lambda x: (x.block_prefix, x.head))],
    }
    return report


def write_prune_report(report: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
