#!/usr/bin/env python
"""Verify the PyTorch/CUDA stack before anything else depends on it.

This is the Phase 1 gate: it proves the selected torch + torchvision pair can do
real work on the GTX 1650, and that CPU and GPU agree numerically. It writes a
machine-readable report so the environment record is measured, not claimed.

Exit codes:
    0  a usable device was verified (GPU, or CPU when CUDA is unavailable)
    1  a required check failed
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

# Redirect caches before importing torch.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fieldlens.env as env  # noqa: E402  (import order is deliberate)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "artifacts" / "reports" / "environment.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the report only")
    args = parser.parse_args(argv)

    failures: list[str] = []
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": None,
        "torchvision": None,
        "cuda_available": False,
        "device": "cpu",
        "device_name": None,
        "device_capability": None,
        "device_memory_bytes": None,
        "device_memory_human": None,
        "gpu_conv2d_ok": False,
        "gpu_autograd_ok": False,
        "cpu_gpu_parity_max_abs_diff": None,
        "checks": [],
    }

    try:
        import torch
        import torchvision
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: cannot import the deep learning stack: {exc}")
        return 1

    report["torch"] = torch.__version__
    report["torchvision"] = torchvision.__version__

    cuda = torch.cuda.is_available()
    report["cuda_available"] = cuda

    if not cuda:
        report["checks"].append(
            {
                "name": "cuda_available",
                "ok": False,
                "detail": "CUDA unavailable; continuing on CPU per the plan's fallback",
            }
        )
    else:
        props = torch.cuda.get_device_properties(0)
        report.update(
            {
                "device": "cuda",
                "device_name": props.name,
                "device_capability": f"{props.major}.{props.minor}",
                "device_memory_bytes": int(props.total_memory),
                "device_memory_human": f"{props.total_memory / 1024**3:.2f} GiB",
            }
        )
        report["checks"].append(
            {"name": "cuda_available", "ok": True, "detail": props.name}
        )

        # A real convolution, not just a tensor allocation.
        try:
            conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).cuda()
            x = torch.randn(2, 3, 32, 32, device="cuda")
            y = conv(x)
            report["gpu_conv2d_ok"] = tuple(y.shape) == (2, 8, 32, 32)
            report["checks"].append(
                {
                    "name": "gpu_conv2d",
                    "ok": bool(report["gpu_conv2d_ok"]),
                    "detail": f"output shape {tuple(y.shape)}",
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"GPU convolution failed: {exc}")
            report["checks"].append(
                {"name": "gpu_conv2d", "ok": False, "detail": str(exc)}
            )

        # A real autograd step.
        try:
            torch.manual_seed(0)
            param = torch.randn(4, 4, device="cuda", requires_grad=True)
            loss = (param @ param).sum()
            loss.backward()
            report["gpu_autograd_ok"] = param.grad is not None and bool(
                torch.isfinite(param.grad).all()
            )
            report["checks"].append(
                {
                    "name": "gpu_autograd",
                    "ok": bool(report["gpu_autograd_ok"]),
                    "detail": "backward produced finite gradients",
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"GPU autograd failed: {exc}")
            report["checks"].append(
                {"name": "gpu_autograd", "ok": False, "detail": str(exc)}
            )

        # CPU/GPU numerical parity on identical inputs.
        try:
            torch.manual_seed(1)
            layer_cpu = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
            layer_gpu = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
            layer_gpu.load_state_dict(layer_cpu.state_dict())
            layer_gpu = layer_gpu.cuda()
            data = torch.randn(4, 3, 64, 64)
            with torch.inference_mode():
                out_cpu = layer_cpu(data)
                out_gpu = layer_gpu(data.cuda()).cpu()
            diff = float((out_cpu - out_gpu).abs().max())
            report["cpu_gpu_parity_max_abs_diff"] = diff
            ok = diff < 1e-3
            report["checks"].append(
                {
                    "name": "cpu_gpu_parity",
                    "ok": ok,
                    "detail": f"max abs difference {diff:.3e} (tolerance 1e-3)",
                }
            )
            if not ok:
                failures.append(f"CPU/GPU parity out of tolerance: {diff:.3e}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"CPU/GPU parity check failed: {exc}")
            report["checks"].append(
                {"name": "cpu_gpu_parity", "ok": False, "detail": str(exc)}
            )

    report["failures"] = failures
    report["disk"] = {
        "project_drive_free": env.format_bytes(env.free_bytes(env.PROJECT_ROOT)),
    }
    if sys.platform == "win32":
        report["disk"]["system_drive_free"] = env.format_bytes(env.free_bytes("C:/"))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("FieldLens AI - stack verification")
        print(f"  torch            : {report['torch']}")
        print(f"  torchvision      : {report['torchvision']}")
        print(f"  cuda available   : {report['cuda_available']}")
        if report["device_name"]:
            print(f"  device           : {report['device_name']} "
                  f"(capability {report['device_capability']}, "
                  f"{report['device_memory_human']})")
        for check in report["checks"]:
            mark = "PASS" if check["ok"] else "FAIL"
            print(f"  [{mark}] {check['name']:<16} {check['detail']}")
        if failures:
            print("\n  FAILURES:")
            for item in failures:
                print(f"    - {item}")
        print(f"\n  report written to {REPORT_PATH}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
