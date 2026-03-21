import numpy as np
import cv2
import os
import time


# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BrownieIdentifier:
    """
    Identifies whether a detected cat is "Brownie" using embedding comparison.

    Backend "dnn": Uses MobileNetV2 ONNX via OpenCV DNN (runs on CPU).
                   Produces 1000-dim embeddings from ImageNet classification logits.
                   Much more discriminative than color histograms.

    Backend "simple": Uses spatial color + texture histograms (252-dim, CPU only).
                      Fast but less discriminative for similar-looking cats.
    """

    def __init__(
        self,
        embedding_path,
        model_path=None,
        backend="dnn",
        similarity_threshold=0.70,
        cache_ttl=10.0,
        standalone=False,
    ):
        self.similarity_threshold = similarity_threshold
        self.backend = backend
        self.cache_ttl = cache_ttl
        self.model_path = model_path
        self.standalone = standalone

        # Tracking cache: tracking_id -> (is_brownie, best_similarity, check_count, timestamp)
        self._cache = {}
        self._max_cache = 500

        # DNN model (loaded lazily)
        self._ort_session = None

        # Load reference embedding
        if embedding_path and os.path.exists(embedding_path):
            self.reference_embedding = np.load(embedding_path).astype(np.float32)
            norm = np.linalg.norm(self.reference_embedding)
            if norm > 0:
                self.reference_embedding = self.reference_embedding / norm
            print(f"Loaded Brownie reference embedding: shape={self.reference_embedding.shape}")

            # Validate dimensions match the backend
            if backend == "dnn":
                if self.reference_embedding.shape[0] != 1000:
                    print(f"WARNING: Reference embedding dim={self.reference_embedding.shape[0]} != dnn backend dim=1000.")
                    print(f"  Recompute with: python3 compute_embeddings.py --photos-dir ~/brownie_photos_cropped/ --backend dnn")
                    self.reference_embedding = None
            elif backend == "simple":
                if self.reference_embedding.shape[0] != 252:
                    print(f"WARNING: Reference embedding dim={self.reference_embedding.shape[0]} != simple backend dim=252.")
                    print(f"  Recompute with: python3 compute_embeddings.py --photos-dir ~/brownie_photos_cropped/ --backend simple")
                    self.reference_embedding = None
        else:
            self.reference_embedding = None
            print("Warning: No reference embedding loaded. Brownie ID will not work.")

        # Initialize DNN backend
        if backend == "dnn":
            self._init_dnn()
        elif backend == "simple":
            print("Brownie identifier using simple (OpenCV histogram) backend.")

    def _init_dnn(self):
        """Load MobileNetV2 ONNX model via ONNX Runtime."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            self.model_path,
            os.path.join(script_dir, "mobilenetv2-12.onnx"),
            os.path.expanduser("~/mobilenetv2-12.onnx"),
        ]
        onnx_path = None
        for c in candidates:
            if c and os.path.exists(c):
                onnx_path = c
                break

        if onnx_path is None:
            print("ERROR: MobileNetV2 ONNX model not found.")
            print("  Download with: python3 download_model.py")
            print("  Falling back to simple backend.")
            self.backend = "simple"
            return

        try:
            import onnxruntime as ort
            self._ort_session = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"],
            )
            self._ort_input_name = self._ort_session.get_inputs()[0].name
            print(f"Brownie identifier using DNN backend (onnxruntime): {os.path.basename(onnx_path)}")
        except ImportError:
            print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
            print("  Falling back to simple backend.")
            self.backend = "simple"
            self._ort_session = None
        except Exception as e:
            print(f"ERROR loading ONNX model: {e}")
            print("  Falling back to simple backend.")
            self.backend = "simple"
            self._ort_session = None

    def close(self):
        """Clean up resources."""
        pass

    def __del__(self):
        self.close()

    # --- Embedding extraction ---

    def get_embedding(self, crop_rgb):
        """Extract features from an image crop."""
        if self.backend == "dnn" and self._ort_session is not None:
            return self._extract_dnn_features(crop_rgb)
        return self._extract_simple_features(crop_rgb)

    def _extract_dnn_features(self, crop_rgb):
        """Extract features using MobileNetV2 ONNX on CPU via ONNX Runtime.

        Produces 1000-dim embedding from ImageNet classification logits.
        These capture high-level visual features (texture, shape, color patterns)
        far better than hand-crafted histograms.
        """
        # Preprocess: resize, normalize with ImageNet stats
        img = cv2.resize(crop_rgb, (224, 224)).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD

        # ONNX Runtime expects NCHW float32
        img_chw = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(img_chw, 0)  # Add batch dim

        output = self._ort_session.run(None, {self._ort_input_name: input_tensor})

        embedding = output[0].flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _extract_simple_features(self, crop_rgb):
        """Extract spatial color + texture features using OpenCV. No model needed."""
        img = cv2.resize(crop_rgb, (128, 128))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        all_features = []
        grid = 3
        cell_h, cell_w = 128 // grid, 128 // grid

        for row in range(grid):
            for col in range(grid):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell = hsv[y1:y2, x1:x2]
                h_hist = cv2.calcHist([cell], [0], None, [12], [0, 180]).flatten()
                s_hist = cv2.calcHist([cell], [1], None, [8], [0, 256]).flatten()
                for h in [h_hist, s_hist]:
                    total = h.sum()
                    if total > 0:
                        h /= total
                all_features.extend([h_hist, s_hist])

        for row in range(grid):
            for col in range(grid):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell = gray[y1:y2, x1:x2]
                gx = cv2.Sobel(cell, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(cell, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(gx ** 2 + gy ** 2)
                angle = np.arctan2(gy, gx) * 180 / np.pi + 180
                grad_hist = np.histogram(angle, bins=8, range=(0, 360), weights=mag)[0].astype(np.float32)
                total = grad_hist.sum()
                if total > 0:
                    grad_hist /= total
                all_features.append(grad_hist)

        features = np.concatenate(all_features)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features.astype(np.float32)

    # --- Identification ---

    def is_brownie(self, crop_rgb):
        """Synchronous check: returns (is_brownie, similarity)."""
        if self.reference_embedding is None:
            return False, 0.0
        embedding = self.get_embedding(crop_rgb)
        similarity = float(np.dot(embedding, self.reference_embedding))
        return similarity >= self.similarity_threshold, similarity

    # --- Tracking cache ---

    def should_check(self, tracking_id):
        """Check multiple frames per track to find the best score.
        Stop checking once identified as Brownie or after max checks."""
        if tracking_id not in self._cache:
            return True
        is_brownie, best_sim, count, ts = self._cache[tracking_id]
        if is_brownie:
            return False
        if count >= 5:
            return False
        return (time.time() - ts) > 2.0

    def cache_result(self, tracking_id, is_brownie, similarity=0.0):
        """Cache identification result, keeping the best similarity score."""
        if len(self._cache) > self._max_cache:
            oldest = sorted(self._cache, key=lambda k: self._cache[k][3])[:100]
            for k in oldest:
                del self._cache[k]
        if tracking_id in self._cache:
            prev_brownie, prev_sim, prev_count, _ = self._cache[tracking_id]
            best_sim = max(prev_sim, similarity)
            is_brownie = is_brownie or prev_brownie or (best_sim >= self.similarity_threshold)
            self._cache[tracking_id] = (is_brownie, best_sim, prev_count + 1, time.time())
        else:
            self._cache[tracking_id] = (is_brownie, similarity, 1, time.time())

    def get_cached_result(self, tracking_id):
        """Get cached result for a tracking ID."""
        if tracking_id in self._cache:
            return self._cache[tracking_id][0]
        return False
