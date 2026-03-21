#!/usr/bin/env python3
"""Download MobileNetV2 ONNX model for Brownie identification."""

import os
import urllib.request
import sys

MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobilenetv2-12.onnx")


def download():
    if os.path.exists(MODEL_FILE):
        size_mb = os.path.getsize(MODEL_FILE) / (1024 * 1024)
        print(f"Model already exists: {MODEL_FILE} ({size_mb:.1f} MB)")
        return

    print(f"Downloading MobileNetV2 ONNX model...")
    print(f"  URL: {MODEL_URL}")
    print(f"  Destination: {MODEL_FILE}")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE, _progress)
        print()
        size_mb = os.path.getsize(MODEL_FILE) / (1024 * 1024)
        print(f"Downloaded: {MODEL_FILE} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        sys.exit(1)


def _progress(count, block_size, total_size):
    pct = count * block_size * 100 // total_size
    print(f"\r  Progress: {pct}%", end="", flush=True)


if __name__ == "__main__":
    download()
