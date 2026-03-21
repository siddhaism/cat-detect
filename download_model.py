#!/usr/bin/env python3
"""Download MobileNetV2 models for Brownie identification.

Usage:
  python3 download_model.py           # Download ONNX model (for CPU/dnn backend)
  python3 download_model.py --hailo   # Compile HEF model (for Hailo backend)
  python3 download_model.py --all     # Download both
"""

import argparse
import os
import subprocess
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ONNX_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
ONNX_FILE = os.path.join(SCRIPT_DIR, "mobilenetv2-12.onnx")
HEF_FILE = os.path.join(SCRIPT_DIR, "mobilenet_v2.hef")


def _progress(count, block_size, total_size):
    if total_size > 0:
        pct = count * block_size * 100 // total_size
        print(f"\r  Progress: {pct}%", end="", flush=True)


def download_onnx():
    """Download MobileNetV2 ONNX model from the ONNX Model Zoo."""
    if os.path.exists(ONNX_FILE):
        size_mb = os.path.getsize(ONNX_FILE) / (1024 * 1024)
        print(f"ONNX model already exists: {ONNX_FILE} ({size_mb:.1f} MB)")
        return True

    print("Downloading MobileNetV2 ONNX model...")
    print(f"  URL: {ONNX_URL}")
    print(f"  Destination: {ONNX_FILE}")

    try:
        urllib.request.urlretrieve(ONNX_URL, ONNX_FILE, _progress)
        print()
        size_mb = os.path.getsize(ONNX_FILE) / (1024 * 1024)
        print(f"Downloaded: {ONNX_FILE} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        if os.path.exists(ONNX_FILE):
            os.remove(ONNX_FILE)
        return False


def compile_hef():
    """Compile MobileNetV2 HEF using the Hailo Model Zoo (hailomz).

    Requires: hailo_model_zoo package (pip install hailo-model-zoo)
    This runs on the Raspberry Pi or any machine with the Hailo DFC installed.
    """
    if os.path.exists(HEF_FILE):
        size_mb = os.path.getsize(HEF_FILE) / (1024 * 1024)
        print(f"HEF model already exists: {HEF_FILE} ({size_mb:.1f} MB)")
        return True

    # Try hailomz compile first (simplest path)
    if _try_hailomz():
        return True

    # Manual compilation path
    print("\nManual HEF compilation:")
    print("  If hailomz is not available, compile manually with the Hailo DFC:")
    print(f"  1. Ensure {ONNX_FILE} exists (run: python3 download_model.py)")
    print("  2. Run on a machine with Hailo DFC installed:")
    print(f"     hailo parser onnx {ONNX_FILE} --end-node-names /Gemm")
    print("     hailo optimize --hw-arch hailo8")
    print(f"     hailo compiler --hw-arch hailo8 -o {HEF_FILE}")
    print()
    print("  Or download a pre-compiled HEF from the Hailo Model Zoo:")
    print("     https://github.com/hailo-ai/hailo_model_zoo")
    return False


def _try_hailomz():
    """Try to compile using hailomz command-line tool."""
    try:
        # Check if hailomz is available
        result = subprocess.run(
            ["hailomz", "--help"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            print("hailomz not available.")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("hailomz not found. Install with: pip install hailo-model-zoo")
        return False

    # Detect Hailo architecture
    hw_arch = "hailo8"
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "HAILO8L" in result.stdout.upper():
            hw_arch = "hailo8l"
        print(f"  Detected architecture: {hw_arch}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  Could not detect architecture, using default: {hw_arch}")

    print(f"Compiling MobileNetV2 HEF for {hw_arch}...")
    try:
        result = subprocess.run(
            [
                "hailomz", "compile",
                "mobilenet_v2",
                f"--hw-arch={hw_arch}",
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for compilation
        )
        if result.returncode == 0:
            # hailomz outputs to current dir or a default location
            # Try to find the output file
            default_output = "mobilenet_v2.hef"
            if os.path.exists(default_output) and default_output != HEF_FILE:
                os.rename(default_output, HEF_FILE)
            elif not os.path.exists(HEF_FILE):
                print(f"  Compilation succeeded but HEF not found at expected path.")
                print(f"  Check hailomz output: {result.stdout}")
                return False
            size_mb = os.path.getsize(HEF_FILE) / (1024 * 1024)
            print(f"Compiled: {HEF_FILE} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  hailomz compile failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  Compilation timed out (>10 min)")
        return False
    except Exception as e:
        print(f"  Compilation error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download MobileNetV2 models")
    parser.add_argument("--hailo", action="store_true", help="Compile HEF for Hailo backend")
    parser.add_argument("--all", action="store_true", help="Download ONNX and compile HEF")
    args = parser.parse_args()

    if args.all:
        ok_onnx = download_onnx()
        print()
        ok_hef = compile_hef()
        if not (ok_onnx or ok_hef):
            sys.exit(1)
    elif args.hailo:
        # Need ONNX first for some compilation paths
        if not os.path.exists(ONNX_FILE):
            download_onnx()
            print()
        if not compile_hef():
            sys.exit(1)
    else:
        if not download_onnx():
            sys.exit(1)


if __name__ == "__main__":
    main()
