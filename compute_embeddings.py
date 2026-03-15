#!/usr/bin/env python3
"""
Compute Brownie's reference embedding from photos.

Usage:
  python3 compute_embeddings.py --photos-dir ~/brownie_photos/
  python3 compute_embeddings.py --photos-dir ~/brownie_photos/ --backend hailo --model resnet_v1_50_h8l.hef
  python3 compute_embeddings.py --photos-dir ~/brownie_photos/ --backend cpu --model mobilenet_v2.onnx

The script processes all images in --photos-dir, extracts features,
averages them, and saves to brownie_embedding.npy.

Recommend 5-10+ photos of Brownie with various angles and lighting.
"""

import argparse
import numpy as np
import cv2
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compute Brownie reference embedding")
    parser.add_argument(
        "--photos-dir", required=True, help="Directory with Brownie photos"
    )
    parser.add_argument(
        "--model", default=None, help="Path to embedding model (.hef or .onnx). Not needed for --backend simple."
    )
    parser.add_argument(
        "--output", default="brownie_embedding.npy", help="Output .npy file"
    )
    parser.add_argument(
        "--backend",
        default="hailo",
        choices=["hailo", "cpu", "simple"],
        help="Feature extraction backend (default: hailo)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Model input size (default: 224)",
    )
    args = parser.parse_args()

    backend = args.backend
    if backend != "simple" and args.model is None:
        print("Error: --model is required for hailo and cpu backends")
        return

    from brownie_identifier import BrownieIdentifier

    # Create identifier without reference embedding (we're building it)
    identifier = BrownieIdentifier(
        embedding_path=None,
        model_path=args.model,
        backend=backend,
    )

    # Process all photos
    photo_dir = Path(args.photos_dir)
    if not photo_dir.exists():
        print(f"Error: directory {photo_dir} does not exist")
        return

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    photos = [p for p in photo_dir.iterdir() if p.suffix.lower() in extensions]

    if not photos:
        print(f"No images found in {photo_dir}")
        return

    print(f"Found {len(photos)} photos in {photo_dir}")
    embeddings = []
    for photo_path in sorted(photos):
        img = cv2.imread(str(photo_path))
        if img is None:
            print(f"  Skipping {photo_path.name} (could not read)")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding = identifier.get_embedding(img_rgb)
        embeddings.append(embedding)
        print(f"  Processed {photo_path.name}: embedding dim={embedding.shape[0]}")

    if not embeddings:
        print("Error: no valid images processed")
        return

    # Average and normalize
    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm

    np.save(args.output, avg_embedding)
    print(
        f"\nSaved reference embedding to {args.output} "
        f"(shape: {avg_embedding.shape}, from {len(embeddings)} photos)"
    )


if __name__ == "__main__":
    main()
