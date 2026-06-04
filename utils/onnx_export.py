"""Export LapAIStudentCNN single-step core to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from lapai_inference.model import LapAIStudentCNN, LapAIStudentConfig


class ONNXCore(nn.Module):
    def __init__(self, inner: LapAIStudentCNN) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, state_in: torch.Tensor) -> torch.Tensor:
        return self.inner(state_in)["pred"]


def export_onnx(model: LapAIStudentCNN, path: Path, opset: int = 17) -> None:
    model.eval()
    wrapped = ONNXCore(model)
    cfg = model.cfg
    dummy = torch.randn(1, cfg.in_channels_raw, cfg.lat, cfg.lon)
    torch.onnx.export(
        wrapped,
        dummy,
        path,
        input_names=["state_in"],
        output_names=["pred"],
        opset_version=opset,
        dynamic_axes={"state_in": {0: "batch"}, "pred": {0: "batch"}},
        dynamo=False,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Export student core to ONNX")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("lapai_forecast.onnx"))
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg")
    cfg = LapAIStudentConfig(**cfg_dict) if cfg_dict else LapAIStudentConfig()
    net = LapAIStudentCNN(cfg)
    net.load_state_dict(ckpt["model"])
    export_onnx(net, args.out, opset=args.opset)
    print("wrote", args.out.resolve())


if __name__ == "__main__":
    main()
