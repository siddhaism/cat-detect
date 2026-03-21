#!/usr/bin/env python3
"""
Crop Brownie photos to just the cat region before computing embeddings.

Usage:
  python3 crop_photos.py --input-dir ~/brownie_photos/ --output-dir ~/brownie_photos_cropped/
  python3 crop_photos.py --input-dir ~/brownie_photos/ --output-dir ~/brownie_photos_cropped/ --padding 0.1

Uses a saliency/contour-based approach to find the main subject (cat) in each photo
and crops tightly around it. Falls back to center-crop if detection fails.
"""

import argparse
import cv2
import numpy as np
import os
from pathlib import Path


def find_subject_bbox(img):
    """Find the bounding box of the main subject using edge/saliency detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Use adaptive thresholding + morphology to find the main blob
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour by area
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Reject if too small (less than 5% of image) or too large (more than 95%)
    img_area = h * w
    if area < img_area * 0.05 or area > img_area * 0.95:
        return None

    return cv2.boundingRect(largest)


def center_crop(img, ratio=0.6):
    """Fallback: crop the center portion of the image."""
    h, w = img.shape[:2]
    new_h, new_w = int(h * ratio), int(w * ratio)
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    return x1, y1, new_w, new_h


def crop_to_subject(img, padding=0.1):
    """Crop image to the main subject with optional padding."""
    h, w = img.shape[:2]

    bbox = find_subject_bbox(img)
    if bbox is None:
        # Fallback to center crop
        bbox = center_crop(img, ratio=0.6)

    x, y, bw, bh = bbox

    # Add padding
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    return img[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser(description="Crop photos to main subject")
    parser.add_argument("--input-dir", required=True, help="Directory with original photos")
    parser.add_argument("--output-dir", required=True, help="Directory for cropped photos")
    parser.add_argument(
        "--padding", type=float, default=0.1,
        help="Padding around detected subject (0.0-0.5, default: 0.1)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    photos = [p for p in input_dir.iterdir() if p.suffix.lower() in extensions]

    if not photos:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(photos)} photos...")
    for photo_path in sorted(photos):
        img = cv2.imread(str(photo_path))
        if img is None:
            print(f"  Skipping {photo_path.name} (could not read)")
            continue

        cropped = crop_to_subject(img, padding=args.padding)
        out_path = output_dir / photo_path.name
        cv2.imwrite(str(out_path), cropped)

        orig_h, orig_w = img.shape[:2]
        crop_h, crop_w = cropped.shape[:2]
        print(f"  {photo_path.name}: {orig_w}x{orig_h} -> {crop_w}x{crop_h}")

    print(f"\nCropped photos saved to {output_dir}")
    print(f"Now run: python3 compute_embeddings.py --photos-dir {output_dir} --backend dnn")


if __name__ == "__main__":
    main()
