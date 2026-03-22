import sys
import heapq
import numpy as np
import cv2
import os
import time
import threading
import queue


# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BrownieIdentifier:
    """
    Identifies whether a detected cat is "Brownie" using embedding comparison.

    Backend "hailo": Uses MobileNetV2 HEF on the Hailo-8 accelerator via HailoRT.
                     Fastest option (~2-5ms per crop). Requires a compiled HEF file
                     and the hailo_platform package. Uses scheduler to share the
                     device with GStreamer's hailonet (YOLO).

    Backend "dnn":   Uses MobileNetV2 ONNX via ONNX Runtime (runs on CPU).
                     Slower (~200-500ms) but no HEF compilation needed.

    Backend "simple": Uses spatial color + texture histograms (252-dim, CPU only).
                      Fast but less discriminative for similar-looking cats.

    All backends run inference in a background thread to avoid blocking
    the GStreamer pipeline callback.
    """

    def __init__(
        self,
        embedding_path,
        model_path=None,
        backend="hailo",
        similarity_threshold=0.58,
    ):
        self.similarity_threshold = similarity_threshold
        self.backend = backend
        self.model_path = model_path

        # Tracking cache: tracking_id -> (is_brownie, scores_list, timestamp)
        self._cache = {}
        self._max_cache = 500

        # Gallery of individual reference embeddings (for max-similarity matching)
        self._gallery_embeddings = None
        self._gallery_colors = None

        # Cross-track spatial persistence
        self._brownie_positions = {}  # track_id -> (cx, cy, timestamp)
        self._spatial_threshold = 0.20  # 20% of frame dimension
        self._spatial_timeout = 3.0  # seconds

        # DNN model (onnxruntime)
        self._ort_session = None
        self._penultimate_output = None

        # Hailo model
        self._hailo_configured = None
        self._hailo_vdevice = None
        self._hailo_input_shape = None
        self._hailo_output_shape = None
        self._hailo_output_dim = 0

        # Color histogram reference (supplementary to DNN/Hailo embedding)
        self._color_reference = None
        self._color_threshold = 0.75  # Minimum color similarity to consider

        # Async inference queue
        self._infer_queue = None
        self._infer_thread = None

        # Load reference embedding
        if embedding_path and os.path.exists(embedding_path):
            self.reference_embedding = np.load(embedding_path).astype(np.float32)
            norm = np.linalg.norm(self.reference_embedding)
            if norm > 0:
                self.reference_embedding = self.reference_embedding / norm
            print(f"Loaded Brownie reference embedding: shape={self.reference_embedding.shape}")
        else:
            self.reference_embedding = None
            print("Warning: No reference embedding loaded. Brownie ID will not work.")

        # Load color reference histogram (if available)
        color_path = embedding_path.replace(".npy", "_color.npy") if embedding_path else None
        if color_path and os.path.exists(color_path):
            self._color_reference = np.load(color_path).astype(np.float32)
            print(f"Loaded Brownie color reference: shape={self._color_reference.shape}")
        else:
            print("No color reference found. Run: python3 compute_embeddings.py --photos-dir ~/brownie_photos_cropped/ --backend <backend>")
            print("  Color histogram will provide supplementary discrimination.")

        # Load gallery embeddings (individual per-photo embeddings for max-similarity)
        if embedding_path:
            gallery_path = embedding_path.replace(".npy", "_gallery.npy")
            if os.path.exists(gallery_path):
                gallery = np.load(gallery_path).astype(np.float32)
                # Normalize each row
                norms = np.linalg.norm(gallery, axis=1, keepdims=True)
                norms[norms == 0] = 1
                self._gallery_embeddings = gallery / norms
                print(f"Loaded Brownie gallery: {self._gallery_embeddings.shape[0]} embeddings")

            gallery_color_path = embedding_path.replace(".npy", "_gallery_color.npy")
            if os.path.exists(gallery_color_path):
                gallery_c = np.load(gallery_color_path).astype(np.float32)
                norms = np.linalg.norm(gallery_c, axis=1, keepdims=True)
                norms[norms == 0] = 1
                self._gallery_colors = gallery_c / norms
                print(f"Loaded Brownie color gallery: {self._gallery_colors.shape[0]} histograms")

        # Initialize backend (with fallback chain: hailo -> dnn -> simple)
        if backend == "hailo":
            self._init_hailo()
            if self.backend != "hailo":
                # Hailo init failed, try dnn fallback
                self._init_dnn()
        elif backend == "dnn":
            self._init_dnn()
        elif backend == "simple":
            print("Brownie identifier using simple (OpenCV histogram) backend.")

        # Validate embedding dimensions match the actual backend after init
        if self.reference_embedding is not None:
            expected_dim = self._get_expected_dim()
            if self.reference_embedding.shape[0] != expected_dim:
                print(f"WARNING: Reference embedding dim={self.reference_embedding.shape[0]} "
                      f"but {self.backend} backend extracts {expected_dim}-dim features.")
                print(f"  Recompute with: python3 compute_embeddings.py --photos-dir ~/brownie_photos_cropped/ --backend {self.backend}")
                self.reference_embedding = None

        # Start background inference thread
        if self._can_infer() and self.reference_embedding is not None:
            self._start_infer_thread()

    def _can_infer(self):
        """Check if the backend is ready for inference."""
        if self.backend == "hailo":
            return self._hailo_configured is not None
        elif self.backend == "dnn":
            return self._ort_session is not None
        return True  # simple backend always works

    def _get_expected_dim(self):
        """Return expected embedding dimension for the current backend config."""
        if self.backend == "hailo":
            return self._hailo_output_dim  # Discovered at init from the HEF
        elif self.backend == "dnn":
            return 1280 if self._penultimate_output else 1000
        elif self.backend == "simple":
            return 252
        return 0

    # --- Hailo backend ---

    def _init_hailo(self):
        """Load an embedding model HEF on the Hailo-8 via hailo_platform.

        Searches for ArcFace or MobileNetV2 HEF files. ArcFace models
        (512-dim embeddings) are preferred as they're designed for identity
        recognition. Uses the HailoRT scheduler to share the device with
        GStreamer's hailonet element (which runs YOLO).
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            self.model_path,
            # ArcFace / embedding models (preferred for identity matching)
            os.path.join(script_dir, "cat_embedding_model.hef"),
            os.path.expanduser("~/repos/cat-feeder/models/cat_embedding_model.hef"),
            os.path.join(script_dir, "arcface_mobilefacenet.hef"),
            "/usr/local/hailo/resources/models/hailo8/arcface_mobilefacenet.hef",
            # MobileNetV2 classification models (fallback)
            os.path.join(script_dir, "mobilenet_v2.hef"),
            os.path.expanduser("~/mobilenet_v2.hef"),
            "/usr/share/hailo-models/mobilenet_v2.hef",
        ]
        hef_path = None
        for c in candidates:
            if c and os.path.exists(c):
                hef_path = c
                break

        if hef_path is None:
            print("No embedding HEF model found. Searched:")
            for c in candidates:
                if c:
                    print(f"  - {c}")
            print("  Copy cat_embedding_model.hef to the repo, or compile MobileNetV2.")
            print("  Falling back to dnn (CPU) backend.")
            self.backend = "dnn"
            return

        try:
            from hailo_platform import (
                HEF,
                VDevice,
                HailoSchedulingAlgorithm,
                FormatType,
            )

            # Load HEF and inspect
            hef = HEF(hef_path)

            # Create VDevice with round-robin scheduler for multi-model sharing.
            # Use group_id so GStreamer's hailonet can share the same VDevice.
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            try:
                params.group_id = "shared"
            except AttributeError:
                pass  # Older HailoRT without group_id support
            self._hailo_vdevice = VDevice(params)

            # Create and configure the inference model
            infer_model = self._hailo_vdevice.create_infer_model(hef_path)

            # Set output to float32 so HailoRT handles dequantization
            infer_model.output().set_format_type(FormatType.FLOAT32)

            self._hailo_configured = infer_model.configure()

            # Discover input/output shapes from the model
            self._hailo_input_shape = infer_model.input().shape
            self._hailo_output_shape = infer_model.output().shape
            # Output dim: flatten the shape (e.g., (512,) -> 512, (1, 1000) -> 1000)
            self._hailo_output_dim = 1
            for d in self._hailo_output_shape:
                self._hailo_output_dim *= d

            print(f"Brownie identifier using HAILO backend ({self._hailo_output_dim}-dim): {os.path.basename(hef_path)}")
            print(f"  Input: {self._hailo_input_shape}, Output: {self._hailo_output_shape}")

        except ImportError:
            print("hailo_platform not installed — Hailo backend unavailable.")
            print("  Falling back to dnn (CPU) backend.")
            self.backend = "dnn"
        except Exception as e:
            print(f"ERROR initializing Hailo backend: {e}")
            print("  Falling back to dnn (CPU) backend.")
            self.backend = "dnn"
            self._hailo_configured = None
            self._hailo_vdevice = None

    def _extract_hailo_features(self, crop_rgb):
        """Extract embedding features on the Hailo-8 accelerator.

        Supports ArcFace (512-dim, 112x112 input) and MobileNetV2 (1000-dim,
        224x224 input). The HEF model includes quantization, so we send uint8
        RGB input. HailoRT handles dequantization of the output to float32.
        """
        # Resize to model input size (typically 224x224)
        input_h = self._hailo_input_shape[0] if len(self._hailo_input_shape) == 3 else self._hailo_input_shape[1]
        input_w = self._hailo_input_shape[1] if len(self._hailo_input_shape) == 3 else self._hailo_input_shape[2]
        img = cv2.resize(crop_rgb, (input_w, input_h))

        # HEF expects uint8 RGB — quantization is baked into the model
        input_data = np.expand_dims(img.astype(np.uint8), 0)

        bindings = self._hailo_configured.create_bindings()
        bindings.input().set_buffer(input_data)

        # Pre-allocate output buffer (required in HailoRT 4.20+)
        output_buffer = np.empty([1] + list(self._hailo_output_shape), dtype=np.float32)
        bindings.output().set_buffer(output_buffer)

        self._hailo_configured.run([bindings], 10000)

        embedding = output_buffer.flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    # --- DNN (onnxruntime) backend ---

    def _init_dnn(self):
        """Load MobileNetV2 ONNX model via ONNX Runtime.

        Attempts to extract penultimate layer features (1280-dim) which are
        much better for visual similarity than classification logits (1000-dim).
        Falls back to logits if the onnx package is not available to inspect
        the model graph.
        """
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

            # Try to find penultimate layer output name for 1280-dim features
            self._penultimate_output = None
            try:
                import onnx
                model = onnx.load(onnx_path)
                for node in reversed(model.graph.node):
                    if node.op_type == "Gemm":
                        self._penultimate_output = node.input[0]
                        break
                if self._penultimate_output:
                    print(f"  Found penultimate layer: {self._penultimate_output} (1280-dim features)")
                del model
            except ImportError:
                print("  Note: 'onnx' package not installed; using 1000-dim logits.")
                print("  For better accuracy, install: pip install onnx")
            except Exception as e:
                print(f"  Could not inspect model graph: {e}; using 1000-dim logits.")

            self._ort_session = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"],
            )
            self._ort_input_name = self._ort_session.get_inputs()[0].name

            feat_dim = 1280 if self._penultimate_output else 1000
            print(f"Brownie identifier using DNN backend (onnxruntime, {feat_dim}-dim): {os.path.basename(onnx_path)}")
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

    def _extract_dnn_features(self, crop_rgb):
        """Extract features using MobileNetV2 ONNX on CPU via ONNX Runtime."""
        img = cv2.resize(crop_rgb, (224, 224)).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img_chw = np.transpose(img, (2, 0, 1))
        input_tensor = np.expand_dims(img_chw, 0)

        if self._penultimate_output:
            output = self._ort_session.run(
                [self._penultimate_output],
                {self._ort_input_name: input_tensor},
            )
        else:
            output = self._ort_session.run(None, {self._ort_input_name: input_tensor})

        embedding = output[0].flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    # --- Simple (histogram) backend ---

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

                # Color histograms (H and S channels)
                cell_hsv = hsv[y1:y2, x1:x2]
                h_hist = cv2.calcHist([cell_hsv], [0], None, [12], [0, 180]).flatten()
                s_hist = cv2.calcHist([cell_hsv], [1], None, [8], [0, 256]).flatten()
                for h in [h_hist, s_hist]:
                    total = h.sum()
                    if total > 0:
                        h /= total
                all_features.extend([h_hist, s_hist])

                # Gradient histogram
                cell_gray = gray[y1:y2, x1:x2]
                gx = cv2.Sobel(cell_gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(cell_gray, cv2.CV_32F, 0, 1, ksize=3)
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

    # --- Color histogram (supplementary to any backend) ---

    @staticmethod
    def compute_color_histogram(crop_rgb):
        """Compute a global HSV color histogram for fur color comparison.

        Uses a global (non-spatial) histogram because bounding box crops
        vary in position/size, making spatial grids unreliable.
        """
        img = cv2.resize(crop_rgb, (64, 64))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # H: 24 bins, S: 12 bins, V: 12 bins = 48 values
        h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [12], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [12], [0, 256]).flatten()
        hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist

    def _color_similarity(self, crop_rgb):
        """Compare crop's color distribution to Brownie's reference colors."""
        if self._color_reference is None:
            return 1.0  # No color ref → don't filter
        hist = self.compute_color_histogram(crop_rgb)
        return float(np.dot(hist, self._color_reference))

    # --- Center crop to reduce background noise ---

    @staticmethod
    def _center_crop(crop_rgb, ratio=0.7):
        """Crop inner portion of bbox to reduce background contamination."""
        h, w = crop_rgb.shape[:2]
        margin_y = int(h * (1 - ratio) / 2)
        margin_x = int(w * (1 - ratio) / 2)
        if margin_y > 0 and margin_x > 0 and h > 2 * margin_y and w > 2 * margin_x:
            return crop_rgb[margin_y:h - margin_y, margin_x:w - margin_x]
        return crop_rgb

    # --- Gallery matching ---

    def _gallery_emb_similarity(self, embedding):
        """Max cosine similarity against all gallery embeddings."""
        if self._gallery_embeddings is not None:
            sims = self._gallery_embeddings @ embedding
            return float(np.max(sims))
        if self.reference_embedding is not None:
            return float(np.dot(embedding, self.reference_embedding))
        return 0.0

    def _gallery_color_similarity(self, crop_rgb):
        """Max cosine similarity against all gallery color histograms."""
        hist = self.compute_color_histogram(crop_rgb)
        if self._gallery_colors is not None:
            sims = self._gallery_colors @ hist
            return float(np.max(sims))
        if self._color_reference is not None:
            return float(np.dot(hist, self._color_reference))
        return 1.0  # No color ref → don't filter

    # --- Cross-track spatial persistence ---

    def register_brownie_position(self, track_id, cx, cy):
        """Record position of a confirmed Brownie track."""
        self._brownie_positions[track_id] = (cx, cy, time.time())
        # Prune old entries
        now = time.time()
        self._brownie_positions = {
            k: v for k, v in self._brownie_positions.items()
            if now - v[2] < self._spatial_timeout
        }

    def inherit_nearby_brownie(self, track_id, cx, cy):
        """Check if a new track is near a recent Brownie and inherit identification."""
        if track_id in self._cache:
            return False  # Already has its own result
        now = time.time()
        for tid, (bx, by, ts) in self._brownie_positions.items():
            if tid == track_id:
                continue
            if now - ts > self._spatial_timeout:
                continue
            dist = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5
            if dist < self._spatial_threshold:
                # Inherit: seed cache as Brownie
                self._cache[track_id] = (True, [self.similarity_threshold + 0.05], now)
                print(f"Brownie inherited: track={track_id} from track={tid} (dist={dist:.3f})", file=sys.stderr)
                return True
        return False

    # --- Async inference (shared by all backends) ---

    def _start_infer_thread(self):
        """Start background thread for async inference."""
        self._infer_queue = queue.Queue(maxsize=4)
        self._infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
        self._infer_thread.start()
        print(f"{self.backend.upper()} inference thread started (non-blocking pipeline mode)")

    def _infer_worker(self):
        """Background thread that processes inference requests."""
        while True:
            try:
                item = self._infer_queue.get(timeout=5.0)
            except queue.Empty:
                continue
            if item is None:
                break
            track_id, crop_rgb = item
            try:
                cropped = self._center_crop(crop_rgb)
                embedding = self.get_embedding(cropped)
                if self.reference_embedding is not None or self._gallery_embeddings is not None:
                    emb_sim = self._gallery_emb_similarity(embedding)
                    color_sim = self._gallery_color_similarity(cropped)
                    # Embedding is the primary discriminator; color is logged but not used in scoring
                    combined = emb_sim
                    self.cache_result(track_id, combined >= self.similarity_threshold, combined)
                    # Debug output
                    if track_id in self._cache:
                        _, scores, _ = self._cache[track_id]
                        avg = sum(scores) / len(scores)
                        cached_result = self._cache[track_id][0]
                        color_str = f", color={color_sim:.3f}" if self._color_reference is not None else ""
                        print(f"Brownie check [{self.backend}]: track={track_id}, emb={emb_sim:.3f}{color_str}, combined={combined:.3f}, avg={avg:.3f} ({len(scores)}x), match={cached_result}", file=sys.stderr)
            except Exception as e:
                print(f"Brownie {self.backend} error: {e}", file=sys.stderr)

    def submit_async(self, tracking_id, crop_rgb):
        """Submit a crop for async inference (non-blocking)."""
        if self._infer_queue is not None:
            try:
                self._infer_queue.put_nowait((tracking_id, crop_rgb))
                return True
            except queue.Full:
                return False
        return False

    def close(self):
        """Clean up resources."""
        if self._infer_queue is not None:
            try:
                self._infer_queue.put_nowait(None)
            except queue.Full:
                pass
        if self._hailo_vdevice is not None:
            try:
                self._hailo_vdevice.release()
            except Exception:
                pass

    def __del__(self):
        self.close()

    # --- Embedding extraction (dispatch) ---

    def get_embedding(self, crop_rgb):
        """Extract features from an image crop (synchronous)."""
        if self.backend == "hailo" and self._hailo_configured is not None:
            return self._extract_hailo_features(crop_rgb)
        elif self.backend == "dnn" and self._ort_session is not None:
            return self._extract_dnn_features(crop_rgb)
        return self._extract_simple_features(crop_rgb)

    # --- Identification ---

    def is_brownie(self, crop_rgb):
        """Synchronous check: returns (is_brownie, similarity)."""
        if self.reference_embedding is None:
            return False, 0.0
        embedding = self.get_embedding(crop_rgb)
        similarity = float(np.dot(embedding, self.reference_embedding))
        return similarity >= self.similarity_threshold, similarity

    # --- Tracking cache ---
    # Cache: tracking_id -> (is_brownie, scores_list, timestamp)
    # Decision uses average score across multiple checks, not just one high score.

    def should_check(self, tracking_id):
        """Check multiple frames per track to build confidence.
        Stop checking once confidently identified or after max checks."""
        if tracking_id not in self._cache:
            return True
        is_brownie, scores, ts = self._cache[tracking_id]
        if is_brownie and len(scores) >= 2:
            return False  # Confidently identified
        if len(scores) >= 5:
            return False  # Max checks reached
        return (time.time() - ts) > 1.0

    def cache_result(self, tracking_id, is_brownie, similarity=0.0):
        """Cache identification result. Decision based on average of all scores.
        Strong matches (threshold + 0.10) are accepted on first check."""
        if len(self._cache) > self._max_cache:
            oldest = heapq.nsmallest(100, self._cache, key=lambda k: self._cache[k][2])
            for k in oldest:
                del self._cache[k]
        if tracking_id in self._cache:
            _, prev_scores, _ = self._cache[tracking_id]
            scores = prev_scores + [similarity]
        else:
            scores = [similarity]
        avg_sim = sum(scores) / len(scores)
        high_conf_threshold = self.similarity_threshold + 0.10
        # Accept on first check if score is very high, otherwise require 2+ checks
        if avg_sim >= high_conf_threshold:
            is_brownie = True
        else:
            is_brownie = avg_sim >= self.similarity_threshold and len(scores) >= 2
        self._cache[tracking_id] = (is_brownie, scores, time.time())

    def get_cached_result(self, tracking_id):
        """Get cached result for a tracking ID."""
        if tracking_id in self._cache:
            return self._cache[tracking_id][0]
        return False
