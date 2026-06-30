#!/usr/bin/env python3
"""Convert anemoi-inference checkpoint (AnemoiModelInterface) to training warm-start format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Bump when the emitted checkpoint structure changes so stale warm-starts regenerate.
# v2: data_indices stored as the 0.14 multi-dataset dict {name: IndexCollection}.
_WARMSTART_FORMAT = 2


def _processor_topology(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Infer processor layer/chunk counts from flat proc.* keys (C4E inference ckpt)."""
    proc_indices: set[int] = set()
    for key in state_dict:
        parts = key.split(".")
        for idx, part in enumerate(parts):
            if part == "proc" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                proc_indices.add(int(parts[idx + 1]))
                break
    n = len(proc_indices) or 1
    return n, n


def _resolve_data_indices(obj, dataset_name: str = "data") -> dict:
    """Return data_indices in the anemoi-training 0.14 multi-dataset dict form.

    0.14's transfer_learning_loading requires hyper_parameters.data_indices to be a
    ``{dataset_name: IndexCollection}`` mapping (each value exposing ``name_to_index``).
    Older C4E inference checkpoints store a single IndexCollection, so wrap it under the
    training dataset name (``data`` for the native_grid dataloader).
    """
    data_indices = obj.data_indices
    if isinstance(data_indices, dict):
        if data_indices and all(hasattr(v, "name_to_index") for v in data_indices.values()):
            return data_indices
        if len(data_indices) == 1:
            return {dataset_name: next(iter(data_indices.values()))}
    return {dataset_name: data_indices}


def _resolve_config(obj, metadata: dict | None) -> object:
    from anemoi.utils.config import DotDict

    if metadata and metadata.get("config") is not None:
        return metadata["config"]
    if getattr(obj, "config", None) is not None:
        return obj.config

    num_layers, num_chunks = _processor_topology(obj.state_dict())
    return DotDict(
        {
            "model": {
                "processor": {
                    "num_layers": num_layers,
                    "num_chunks": num_chunks,
                }
            }
        }
    )


def _existing_warmstart_format(path: Path) -> int:
    """Read the warm-start format stamp without loading tensors (defaults to 1=legacy)."""
    try:
        meta = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return -1
    if isinstance(meta, dict):
        return int(meta.get("lapai_warmstart_format", 1))
    return -1


def convert_inference_to_warmstart(in_path: Path, out_path: Path, *, force: bool = False) -> Path:
    """Build a PyTorch Lightning-style checkpoint for anemoi-training transfer learning."""
    in_path = in_path.resolve()
    out_path = out_path.resolve()

    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    if (
        out_path.is_file()
        and not force
        and out_path.stat().st_mtime >= in_path.stat().st_mtime
        and _existing_warmstart_format(out_path) == _WARMSTART_FORMAT
    ):
        return out_path

    obj = torch.load(in_path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and "state_dict" in obj:
        if out_path != in_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(obj, out_path)
        return out_path

    from anemoi.models.interface import AnemoiModelInterface
    from anemoi.utils.checkpoints import has_metadata, load_metadata

    if not isinstance(obj, AnemoiModelInterface):
        raise TypeError(f"Expected AnemoiModelInterface or PL checkpoint dict, got {type(obj)}")

    metadata = load_metadata(str(in_path)) if has_metadata(str(in_path)) else None
    config = _resolve_config(obj, metadata)
    data_indices = _resolve_data_indices(obj)

    ckpt = {
        "state_dict": {f"model.{k}": v for k, v in obj.state_dict().items()},
        "hyper_parameters": {
            "config": config,
            "data_indices": data_indices,
        },
        "pytorch-lightning_version": "2.4.0",
        "lapai_warmstart_format": _WARMSTART_FORMAT,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    return out_path


def default_warmstart_path(inference_path: Path) -> Path:
    if inference_path.name == "inference.ckpt":
        return inference_path.with_name("warmstart.ckpt")
    return inference_path.with_name(f"{inference_path.stem}_warmstart.ckpt")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", type=Path, required=True, help="inference.ckpt path")
    p.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=None,
        help="Output warm-start path (default: sibling warmstart.ckpt)",
    )
    p.add_argument("--force", action="store_true", help="Regenerate even if output is newer")
    args = p.parse_args()

    out_path = args.out_path or default_warmstart_path(args.in_path)
    result = convert_inference_to_warmstart(args.in_path, out_path, force=args.force)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
