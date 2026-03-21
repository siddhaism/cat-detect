import numpy as np
import cv2
import os
import time
import multiprocessing as mp


def _hailo_worker(model_path, input_queue, result_queue):
    """Worker process that runs the Hailo embedding model."""
    from hailo_platform import VDevice, HEF, InferVStreams, InputVStreamParams, OutputVStreamParams

    hef = HEF(model_path)
    params = VDevice.create_params()
    vdevice = VDevice(params)
    network_group = vdevice.configure(hef)[0]
    activated = network_group.activate()
    activated.__enter__()

    input_info = hef.get_input_vstream_infos()[0]
    input_name = input_info.name
    input_shape = input_info.shape  # (H, W, C)

    input_vstream_params = InputVStreamParams.make(network_group)
    output_vstream_params = OutputVStreamParams.make(network_group)

    print(f"[Hailo worker] Ready: input={input_name} {input_shape}")

    try:
        while True:
            item = input_queue.get()
            if item is None:
                break
            track_id, crop_rgb = item
            try:
                img = cv2.resize(crop_rgb, (input_shape[1], input_shape[0])).astype(np.uint8)
                input_data = np.expand_dims(img, 0)

                with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer:
                    result = infer.infer({input_name: input_data})

                output_key = list(result.keys())[0]
                embedding = result[output_key].flatten().astype(np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                result_queue.put((track_id, embedding))
            except Exception as e:
                print(f"[Hailo worker] inference error: {e}")
                result_queue.put((track_id, None))
    finally:
        try:
            activated.__exit__(None, None, None)
        except Exception:
            pass


class BrownieIdentifier:
    """
    Identifies whether a detected cat is "Brownie" using embedding comparison.
    Runs the embedding model on Hailo in a separate process to avoid device conflicts.
    """

    def __init__(
        self,
        embedding_path,
        model_path,
        backend="hailo",
        similarity_threshold=0.70,
        cache_ttl=10.0,
        standalone=False,
    ):
        self.similarity_threshold = similarity_threshold
        self.backend = backend
        self.cache_ttl = cache_ttl
        self.model_path = model_path
        self.standalone = standalone

        # Tracking cache: tracking_id -> (is_brownie, similarity, timestamp)
        self._cache = {}
        self._max_cache = 500

        # Load reference embedding
        if embedding_path and os.path.exists(embedding_path):
            self.reference_embedding = np.load(embedding_path)
            norm = np.linalg.norm(self.reference_embedding)
            if norm > 0:
                self.reference_embedding = self.reference_embedding / norm
            print(f"Loaded Brownie reference embedding: shape={self.reference_embedding.shape}")

            # Validate dimensions match the backend (skip if hailo — may fall back later)
            if backend == "simple":
                # 3x3 grid * (12 H-bins + 8 S-bins) + 3x3 grid * 8 gradient bins = 9*20 + 9*8 = 252
                expected_dim = 252
                if self.reference_embedding.shape[0] != expected_dim:
                    print(f"WARNING: Reference embedding dim={self.reference_embedding.shape[0]} != simple backend dim={expected_dim}.")
                    print(f"  Recompute with: python3 compute_embeddings.py --photos-dir ~/brownie_photos_cropped/ --backend simple")
                    self.reference_embedding = None

            # Store reference photos dir for possible re-computation on fallback
            self._photos_dir = None
            for candidate in [
                os.path.expanduser("~/brownie_photos_cropped"),
                os.path.expanduser("~/brownie_photos"),
            ]:
                if os.path.isdir(candidate):
                    self._photos_dir = candidate
                    break
        else:
            self.reference_embedding = None
            print("Warning: No reference embedding loaded. Brownie ID will not work.")

        # Initialize backend
        self._worker = None
        self._input_queue = None
        self._result_queue = None
        self._hailo_direct = None
        self._hailo_lazy = False

        if backend == "hailo":
            if standalone:
                # Defer Hailo init to first use so hailonet can claim the device first
                self._hailo_lazy = True
                print("Hailo embedding model will initialize on first use (deferred).")
            else:
                self._start_hailo_worker()
        elif backend == "simple":
            print("Brownie identifier using simple (OpenCV histogram) backend.")

    def _recompute_simple_embedding(self):
        """Recompute reference embedding using simple backend from reference photos."""
        if not self._photos_dir:
            print("No reference photos directory found. Brownie ID disabled.")
            self.reference_embedding = None
            return
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        from pathlib import Path
        photos = [p for p in Path(self._photos_dir).iterdir() if p.suffix.lower() in extensions]
        if not photos:
            print(f"No photos in {self._photos_dir}. Brownie ID disabled.")
            self.reference_embedding = None
            return
        embeddings = []
        for p in sorted(photos):
            img = cv2.imread(str(p))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embeddings.append(self._extract_simple_features(img_rgb))
        if embeddings:
            avg = np.mean(embeddings, axis=0).astype(np.float32)
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg = avg / norm
            self.reference_embedding = avg
            print(f"Recomputed simple embedding from {len(embeddings)} photos (dim={avg.shape[0]})")
        else:
            self.reference_embedding = None
            print("Failed to recompute embedding. Brownie ID disabled.")

    def _init_hailo_direct(self):
        """Initialize Hailo directly in this process (for standalone/compute_embeddings use)."""
        from hailo_platform import VDevice, HEF, InferVStreams, InputVStreamParams, OutputVStreamParams

        self._hailo_imports = {
            "InferVStreams": InferVStreams,
            "InputVStreamParams": InputVStreamParams,
            "OutputVStreamParams": OutputVStreamParams,
        }

        self.hef = HEF(self.model_path)
        params = VDevice.create_params()
        # Enable round-robin scheduling for device sharing with hailonet in GStreamer pipeline
        try:
            from hailo_platform import HailoSchedulingAlgorithm
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            print("Hailo embedding model using ROUND_ROBIN scheduling for device sharing")
        except (ImportError, AttributeError):
            pass
        self.vdevice = VDevice(params)
        self.network_group = self.vdevice.configure(self.hef)[0]
        self._activated_ng = self.network_group.activate()
        self._activated_ng.__enter__()

        input_info = self.hef.get_input_vstream_infos()[0]
        self._input_name = input_info.name
        self._input_shape = input_info.shape
        self._hailo_direct = True

        print(f"Hailo embedding model loaded (direct): input={self._input_name} {self._input_shape}")

    def _start_hailo_worker(self):
        """Start the Hailo inference worker process."""
        self._input_queue = mp.Queue(maxsize=4)
        self._result_queue = mp.Queue(maxsize=4)
        self._worker = mp.Process(
            target=_hailo_worker,
            args=(self.model_path, self._input_queue, self._result_queue),
            daemon=True,
        )
        self._worker.start()
        # Wait briefly for worker to initialize
        time.sleep(2)
        if not self._worker.is_alive():
            print("Warning: Hailo worker process died. Falling back to simple backend.")
            self.backend = "simple"
        else:
            print("Hailo embedding worker process started.")

    def close(self):
        """Shut down the worker process or direct Hailo resources."""
        if self._hailo_direct and hasattr(self, "_activated_ng"):
            try:
                self._activated_ng.__exit__(None, None, None)
            except Exception:
                pass
        if self._input_queue is not None:
            try:
                self._input_queue.put(None)
            except Exception:
                pass
        if self._worker is not None:
            self._worker.join(timeout=3)
            if self._worker.is_alive():
                self._worker.terminate()

    def __del__(self):
        self.close()

    # --- Embedding extraction ---

    def get_embedding(self, crop_rgb):
        """Extract features from an image crop."""
        if self.backend == "simple":
            return self._extract_simple_features(crop_rgb)
        # Lazy init: create Hailo VDevice on first use (after hailonet has started)
        if self._hailo_lazy and not self._hailo_direct:
            print("Initializing Hailo embedding model (deferred)...", flush=True)
            try:
                self._init_hailo_direct()
            except Exception as e:
                print(f"Hailo embedding init failed: {e}", flush=True)
                print("Falling back to simple (histogram) backend.", flush=True)
                self.backend = "simple"
                self._recompute_simple_embedding()
                return self._extract_simple_features(crop_rgb)
            self._hailo_lazy = False
        if self._hailo_direct:
            # Direct Hailo inference (standalone mode)
            InferVStreams = self._hailo_imports["InferVStreams"]
            InputVStreamParams = self._hailo_imports["InputVStreamParams"]
            OutputVStreamParams = self._hailo_imports["OutputVStreamParams"]

            img = cv2.resize(crop_rgb, (self._input_shape[1], self._input_shape[0])).astype(np.uint8)
            input_data = np.expand_dims(img, 0)

            input_vstream_params = InputVStreamParams.make(self.network_group)
            output_vstream_params = OutputVStreamParams.make(self.network_group)

            with InferVStreams(self.network_group, input_vstream_params, output_vstream_params) as infer:
                result = infer.infer({self._input_name: input_data})

            output_key = list(result.keys())[0]
            embedding = result[output_key].flatten().astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        else:
            raise RuntimeError("Use submit_crop/poll_results for worker mode")

    def _extract_simple_features(self, crop_rgb):
        """Extract spatial color + texture features using OpenCV. No model needed.

        Uses a 3x3 grid of color histograms (captures WHERE colors appear)
        plus LBP texture features to distinguish cats with similar coloring
        but different patterns/markings.
        """
        img = cv2.resize(crop_rgb, (128, 128))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        all_features = []

        # Spatial color histograms: 3x3 grid
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

        # LBP-like texture features per 3x3 grid cell
        for row in range(grid):
            for col in range(grid):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell = gray[y1:y2, x1:x2]

                # Compute simple gradient magnitudes as texture descriptor
                gx = cv2.Sobel(cell, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(cell, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(gx ** 2 + gy ** 2)
                angle = np.arctan2(gy, gx) * 180 / np.pi + 180  # 0-360

                # Gradient orientation histogram (8 bins)
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

    # --- Identification (async via worker) ---

    def submit_crop(self, tracking_id, crop_rgb):
        """Send a crop to the Hailo worker for async identification."""
        if self._input_queue is not None:
            try:
                self._input_queue.put_nowait((tracking_id, crop_rgb))
            except Exception:
                pass  # Queue full, skip this frame

    def poll_results(self):
        """Check for completed identification results from the worker."""
        if self._result_queue is None:
            return
        while not self._result_queue.empty():
            try:
                track_id, embedding = self._result_queue.get_nowait()
                if embedding is not None and self.reference_embedding is not None:
                    similarity = float(np.dot(embedding, self.reference_embedding))
                    is_brownie = similarity >= self.similarity_threshold
                    self.cache_result(track_id, is_brownie)
                    if is_brownie:
                        print(f"Brownie identified! (track={track_id}, sim={similarity:.3f})")
                else:
                    self.cache_result(track_id, False)
            except Exception:
                break

    # --- Identification (sync for simple backend) ---

    def is_brownie(self, crop_rgb):
        """Synchronous check using any backend (simple or standalone Hailo)."""
        if self.reference_embedding is None:
            return False, 0.0
        embedding = self.get_embedding(crop_rgb)
        similarity = float(np.dot(embedding, self.reference_embedding))
        return similarity >= self.similarity_threshold, similarity

    # --- Tracking cache ---

    def should_check(self, tracking_id):
        """Rate-limit: only check once per tracking ID, or re-check after TTL."""
        if tracking_id not in self._cache:
            return True
        _, ts = self._cache[tracking_id]
        return (time.time() - ts) > self.cache_ttl

    def cache_result(self, tracking_id, is_brownie):
        """Cache identification result for a tracking ID."""
        if len(self._cache) > self._max_cache:
            oldest = sorted(self._cache, key=lambda k: self._cache[k][1])[:100]
            for k in oldest:
                del self._cache[k]
        self._cache[tracking_id] = (is_brownie, time.time())

    def get_cached_result(self, tracking_id):
        """Get cached result for a tracking ID."""
        if tracking_id in self._cache:
            return self._cache[tracking_id][0]
        return False
