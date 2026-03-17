#!/usr/bin/env python3
"""Quick diagnostic for Brownie identification."""
import numpy as np
import cv2
import os
import sys

print("=== Brownie Identifier Diagnostic ===\n")

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
        model_path="",
        backend="simple",
        similarity_threshold=0.30,
        standalone=True,
    )
    if bi.reference_embedding is not None:
        print(f"[OK] BrownieIdentifier created (ref dim={bi.reference_embedding.shape[0]})")
    else:
        print("[FAIL] reference_embedding is None!")
        sys.exit(1)
except Exception as e:
    print(f"[FAIL] Creation failed: {e}")
    sys.exit(1)

# Test 4: Test against Brownie photos
photo_dir = os.path.expanduser("~/brownie_photos")
if not os.path.exists(photo_dir):
    print(f"\n[FAIL] Photo directory not found: {photo_dir}")
    sys.exit(1)

photos = sorted([f for f in os.listdir(photo_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"\n=== Testing against {len(photos)} Brownie photos ===")
for photo in photos:
    path = os.path.join(photo_dir, photo)
    img = cv2.imread(path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    is_brownie, similarity = bi.is_brownie(img_rgb)
    status = "BROWNIE" if is_brownie else "not brownie"
    print(f"  {photo:20s} sim={similarity:.3f}  -->  {status}")

print("\nIf most photos above show 'BROWNIE', the identifier works.")
print("If they show 'not brownie', the threshold needs lowering or the approach needs changing.")
