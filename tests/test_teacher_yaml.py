"""Tests for ``utils.teacher_yaml`` line parser."""

from __future__ import annotations

from pathlib import Path

from utils.teacher_yaml import parse_teacher_aifs_yaml_file, teacher_checkpoint_resolved


def test_parse_skips_null_revision(tmp_path):
    cfg = tmp_path / "teacher_aifs.yaml"
    cfg.write_text(
        "revision: null\n"
        "huggingface_repo_id: ecmwf/x\n"
        "local_checkpoint_relative: models/a.ckpt\n",
        encoding="ascii",
    )
    pins = parse_teacher_aifs_yaml_file(cfg)
    assert "revision" not in pins
    assert pins["huggingface_repo_id"] == "ecmwf/x"


def test_teacher_checkpoint_resolved(tmp_path):
    cfg = tmp_path / "configs" / "teacher_aifs.yaml"
    cfg.parent.mkdir(parents=True)
    ck = tmp_path / "models" / "a.ckpt"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x")
    cfg.write_text("local_checkpoint_relative: models/a.ckpt\n", encoding="ascii")
    out = teacher_checkpoint_resolved(tmp_path, cfg_path=cfg)
    assert out == ck.resolve()
