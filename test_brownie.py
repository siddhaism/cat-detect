#!/usr/bin/env python3
"""Quick diagnostic for Brownie identification.

Usage:
  python3 test_brownie.py                    # Auto-detect backend (hailo -> dnn -> simple)
  python3 test_brownie.py --backend hailo    # Force Hailo backend
  python3 test_brownie.py --backend dnn      # Force CPU/onnxruntime backend
"""
import argparse
import numpy as np
import cv2
import os
import sys

parser = argparse.ArgumentParser(description="Brownie identifier diagnostic")
parser.add_argument("--backend", default="hailo", choices=["hailo", "dnn", "simple"],
                    help="Backend to test (default: hailo)")
parser.add_argument("--threshold", type=float, default=0.40,
                    help="Similarity threshold (default: 0.40)")
parser.add_argument("--photos-dir", default=os.path.expanduser("~/brownie_photos"),
                    help="Directory with Brownie test photos")
args = parser.parse_args()

print(f"=== Brownie Identifier Diagnostic (backend={args.backend}) ===\n")

# Test 1: Import
try:
    from brownie_identifier import BrownieIdentifier
    print("[OK] BrownieIdentifier imported")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Embedding file
embedding_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brownie_embedding.npy")
print(f"\nEmbedding path: {embedding_path}")
print(f"Exists: {os.path.exists(embedding_path)}")
if os.path.exists(embedding_path):
    e = np.load(embedding_path)
    print(f"Embedding dim: {e.shape[0]}")
else:
    print("[FAIL] Embedding file missing!")
    sys.exit(1)

# Test 3: Create identifier
try:
    bi = BrownieIdentifier(
        embedding_path=embedding_path,
        backend=args.backend,
        similarity_threshold=args.threshold,
    )
    print(f"[OK] BrownieIdentifier created (backend={bi.backend}, "
          f"ref dim={bi.reference_embedding.shape[0] if bi.reference_embedding is not None else 'None'})")
    if bi.reference_embedding is None:
        print("[FAIL] reference_embedding is None! Dimension mismatch or model not found.")
        print(f"  Recompute with: python3 compute_embeddings.py --photos-dir ~/brownie_photos_cropped/ --backend {bi.backend}")
        sys.exit(1)
except Exception as e:
    print(f"[FAIL] Creation failed: {e}")
    sys.exit(1)

# Test 4: Test against Brownie photos
if not os.path.exists(args.photos_dir):
    print(f"\n[SKIP] Photo directory not found: {args.photos_dir}")
    print("  Pass --photos-dir to specify a different directory.")
    sys.exit(0)

extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
photos = sorted([f for f in os.listdir(args.photos_dir) if f.lower().endswith(extensions)])
print(f"\n=== Testing against {len(photos)} photos (backend={bi.backend}) ===")
for photo in photos:
    path = os.path.join(args.photos_dir, photo)
    img = cv2.imread(path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    is_brownie, similarity = bi.is_brownie(img_rgb)
    status = "BROWNIE" if is_brownie else "not brownie"
    print(f"  {photo:20s} sim={similarity:.3f}  -->  {status}")

print(f"\nBackend: {bi.backend} | Threshold: {args.threshold}")
print("If most photos show 'BROWNIE', the identifier works.")
