"""Copy A2 full yaml/launcher to Cassava with Unix LF, no BOM."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    (ROOT / "configs" / "trackA_prune_full.yaml", "/local/Mthetho/lapai-forecast/configs/trackA_prune_full.yaml"),
    (ROOT / "scripts" / "run_a2_prune_full.sh", "/local/Mthetho/lapai-forecast/scripts/run_a2_prune_full.sh"),
    (ROOT / "scripts" / "run_a2_prune_full.sh", "/local/Mthetho/run_a2_prune_full.sh"),
    (ROOT / "scripts" / "run_a2_forecast_gate_full.sh", "/local/Mthetho/lapai-forecast/scripts/run_a2_forecast_gate_full.sh"),
    (ROOT / "scripts" / "run_a2_forecast_gate_full.sh", "/local/Mthetho/run_a2_forecast_gate_full.sh"),
]


def main() -> None:
    for src, dst in PAIRS:
        data = src.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if data.startswith(b"\xef\xbb\xbf"):
            data = data[3:]
        cr = data.count(b"\r")
        lf = data.count(b"\n")
        print(f"put {src.name} -> {dst} bytes={len(data)} cr={cr} lf={lf}")
        r = subprocess.run(["ssh", "cassava-gpu", f"cat > {dst}"], input=data)
        if r.returncode != 0:
            raise SystemExit(f"failed writing {dst} rc={r.returncode}")
    print("ok")


if __name__ == "__main__":
    main()
