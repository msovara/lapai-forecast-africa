"""Smoke tests for student forward and ONNX wrapper."""

import torch

from lapai_inference.model import LapAIStudentCNN, LapAIStudentConfig
from utils.onnx_export import ONNXCore, export_onnx


def test_student_forward_shapes():
    cfg = LapAIStudentConfig(in_channels_raw=65, lat=32, lon=64)
    net = LapAIStudentCNN(cfg)
    x = torch.randn(2, cfg.in_channels_raw, cfg.lat, cfg.lon)
    out = net(x)
    assert out["pred"].shape == (2, cfg.out_channels, cfg.lat, cfg.lon)


def test_onnx_export_tmp_path(tmp_path):
    cfg = LapAIStudentConfig(in_channels_raw=65, lat=17, lon=18, base_channels=64, stages=(1, 1, 1))
    net = LapAIStudentCNN(cfg)
    onnx_p = tmp_path / "m.onnx"
    export_onnx(net, onnx_p, opset=17)
    assert onnx_p.exists()
    wrapped = ONNXCore(net)
    wrapped.eval()
    x = torch.randn(1, cfg.in_channels_raw, cfg.lat, cfg.lon)
    with torch.no_grad():
        y = wrapped(x)
    assert y.shape == (1, cfg.out_channels, cfg.lat, cfg.lon)
