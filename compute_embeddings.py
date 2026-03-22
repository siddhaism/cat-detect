#!/usr/bin/env python3
"""
Compute Brownie's reference embedding from photos.

Usage:
  python3 compute_embeddings.py --photos-dir ~/brownie_photos_cropped/ --backend hailo
  python3 compute_embeddings.py --photos-dir ~/brownie_photos_cropped/ --backend dnn
  python3 compute_embeddings.py --photos-dir ~/brownie_photos_cropped/ --backend simple

The script processes all images in --photos-dir, extracts features,
averages them, and saves to brownie_embedding.npy.

Recommend 5-10+ photos of Brownie with various angles and lighting.
For best results, use cropped photos (just the cat, no background):
  python3 crop_photos.py --input-dir ~/brownie_photos/ --output-dir ~/brownie_photos_cropped/
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
        "--model", default=None, help="Path to model file (ONNX for dnn, HEF for hailo)"
    )
    parser.add_argument(
        "--output", default="brownie_embedding.npy", help="Output .npy file"
    )
    parser.add_argument(
        "--backend",
        default="hailo",
        choices=["hailo", "dnn", "simple"],
        help="Feature extraction backend (default: hailo)",
    )
    args = parser.parse_args()

    from brownie_identifier import BrownieIdentifier

    identifier = BrownieIdentifier(
        embedding_path=None,
        model_path=args.model,
        backend=args.backend,
    )

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

    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm

    np.save(args.output, avg_embedding)
    print(
        f"\nSaved reference embedding to {args.output} "
        f"(shape: {avg_embedding.shape}, from {len(embeddings)} photos)"
    )

    # Save gallery (individual embeddings for max-similarity matching)
    gallery = np.array(embeddings, dtype=np.float32)
    gallery_output = args.output.replace(".npy", "_gallery.npy")
    np.save(gallery_output, gallery)
    print(f"Saved embedding gallery to {gallery_output} (shape: {gallery.shape})")

    # Also compute and save a color histogram reference
    color_hists = []
    for photo_path in sorted(photos):
        img = cv2.imread(str(photo_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = BrownieIdentifier.compute_color_histogram(img_rgb)
        color_hists.append(hist)

    if color_hists:
        avg_color = np.mean(color_hists, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg_color)
        if norm > 0:
            avg_color = avg_color / norm
        color_output = args.output.replace(".npy", "_color.npy")
        np.save(color_output, avg_color)
        print(f"Saved color reference to {color_output} (shape: {avg_color.shape})")

        # Save color gallery
        gallery_color = np.array(color_hists, dtype=np.float32)
        gallery_color_output = args.output.replace(".npy", "_gallery_color.npy")
        np.save(gallery_color_output, gallery_color)
        print(f"Saved color gallery to {gallery_color_output} (shape: {gallery_color.shape})")

    # Compute and save LBP texture features
    def _save_feature(compute_fn, name):
        features = []
        for photo_path in sorted(photos):
            img = cv2.imread(str(photo_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feat = compute_fn(img_rgb)
            features.append(feat)
        if features:
            avg = np.mean(features, axis=0).astype(np.float32)
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg /= norm
            ref_path = args.output.replace(".npy", f"_{name}.npy")
            np.save(ref_path, avg)
            print(f"Saved {name} reference to {ref_path} (shape: {avg.shape})")
            gallery_arr = np.array(features, dtype=np.float32)
            gal_path = args.output.replace(".npy", f"_gallery_{name}.npy")
            np.save(gal_path, gallery_arr)
            print(f"Saved {name} gallery to {gal_path} (shape: {gallery_arr.shape})")

    _save_feature(BrownieIdentifier.compute_lbp_features, "texture")
    _save_feature(BrownieIdentifier.compute_shape_features, "shape")
    _save_feature(BrownieIdentifier.compute_edge_features, "edge")


if __name__ == "__main__":
    main()
