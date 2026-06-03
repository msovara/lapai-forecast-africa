"""Parse ``configs/teacher_aifs.yaml`` without pulling in PyYAML."""

from __future__ import annotations

import os
from pathlib import Path

# Default teacher config; override with env LAPAI_TEACHER_CONFIG (e.g. configs/teacher_n320_gt6.yaml).
DEFAULT_TEACHER_CONFIG = "configs/teacher_aifs.yaml"
TEACHER_CONFIG_ENV = "LAPAI_TEACHER_CONFIG"


def resolve_teacher_config_path(repo_root: Path, cli_value: str | Path | None = None) -> Path:
    """Pick the teacher YAML: CLI value > env LAPAI_TEACHER_CONFIG > default. Relative paths join repo_root."""
    chosen = cli_value or os.environ.get(TEACHER_CONFIG_ENV) or DEFAULT_TEACHER_CONFIG
    path = Path(chosen)
    return path if path.is_absolute() else (repo_root / path)

_TEACHER_KEYS = frozenset(
    {
        "huggingface_repo_id",
        "checkpoint_filename",
        "revision",
        "local_checkpoint_relative",
    }
)


def parse_teacher_aifs_yaml_file(cfg_path: Path) -> dict[str, str]:
    """Colon-style key/value lines only; ignores YAML blocks."""
    out: dict[str, str] = {}
    if not cfg_path.is_file():
        return out

    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or ":" not in stripped:
            continue
        raw_key, rest = stripped.split(":", 1)
        key = raw_key.strip()
        if key not in _TEACHER_KEYS:
            continue
        val = rest.strip().strip("\"'")
        if val.lower() not in ("", "null", "~", "none"):
            out[key] = val

    return out


def teacher_checkpoint_resolved(repo_root: Path, cfg_path: Path | None = None) -> Path | None:
    cfg = cfg_path or (repo_root / "configs" / "teacher_aifs.yaml")
    rel = parse_teacher_aifs_yaml_file(cfg).get("local_checkpoint_relative")
    if not rel:
        return None
    return (repo_root / rel).resolve()
