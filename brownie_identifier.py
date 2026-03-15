import numpy as np
import cv2
import os
import time


class BrownieIdentifier:
    """
    Identifies whether a detected cat is "Brownie" using embedding comparison.
    Supports Hailo (primary) and ONNX/CPU (fallback) backends.
    """

    def __init__(
        self,
        embedding_path,
        model_path,
        backend="hailo",
        similarity_threshold=0.70,
        input_size=(224, 224),
        cache_ttl=10.0,
    ):
        self.similarity_threshold = similarity_threshold
        self.input_size = input_size
        self.backend = backend
        self.cache_ttl = cache_ttl

        # Tracking cache: tracking_id -> (is_brownie, timestamp)
        self._cache = {}
        self._max_cache = 500

        # Load reference embedding
        if embedding_path and os.path.exists(embedding_path):
            self.reference_embedding = np.load(embedding_path)
            self.reference_embedding = self.reference_embedding / np.linalg.norm(
                self.reference_embedding
            )
            print(
                f"Loaded Brownie reference embedding: shape={self.reference_embedding.shape}"
            )
        else:
            self.reference_embedding = None
            print("Warning: No reference embedding loaded. Brownie ID will not work.")

        # Initialize backend
        if backend == "hailo":
            self._init_hailo(model_path)
        else:
            self._init_cpu(model_path)

    def _init_hailo(self, model_path):
        """Initialize Hailo VDevice and configure network."""
        from hailo_platform import (
            VDevice,
            HEF,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
        )

        self._hailo_imports = {
            "InferVStreams": InferVStreams,
            "InputVStreamParams": InputVStreamParams,
            "OutputVStreamParams": OutputVStreamParams,
        }

        self.hef = HEF(model_path)
        params = VDevice.create_params()
        self.vdevice = VDevice(params)
        self.network_group = self.vdevice.configure(self.hef)[0]

        # Activate the network group (required before inference)
        self._activated_ng = self.network_group.activate()
        self._activated_ng.__enter__()

        # Get input/output info
        input_vstreams_info = self.hef.get_input_vstream_infos()
        output_vstreams_info = self.hef.get_output_vstream_infos()
        self._input_name = input_vstreams_info[0].name
        self._input_shape = input_vstreams_info[0].shape
        self._output_name = output_vstreams_info[0].name

        print(
            f"Hailo embedding model loaded: input={self._input_name} {self._input_shape}, "
            f"output={self._output_name}"
        )

    def _init_cpu(self, model_path):
        """Initialize ONNX Runtime session for CPU fallback."""
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        self._input_shape = self.session.get_inputs()[0].shape
        print(f"ONNX CPU embedding model loaded: input={self._input_name} {self._input_shape}")

    def preprocess_hailo(self, crop_rgb):
        """Resize crop for Hailo input (uint8, no normalization — HEF handles it)."""
        img = cv2.resize(crop_rgb, self.input_size)
        return img.astype(np.uint8)

    def preprocess_cpu(self, crop_rgb):
        """Resize and normalize crop for ONNX/CPU input (float32, ImageNet normalization)."""
        img = cv2.resize(crop_rgb, self.input_size)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return img

    def get_embedding(self, crop_rgb):
        """Run the embedding model and return L2-normalized feature vector."""
        if self.backend == "hailo":
            preprocessed = self.preprocess_hailo(crop_rgb)
            return self._infer_hailo(preprocessed)
        else:
            preprocessed = self.preprocess_cpu(crop_rgb)
            return self._infer_cpu(preprocessed)

    def _infer_hailo(self, preprocessed):
        InferVStreams = self._hailo_imports["InferVStreams"]
        InputVStreamParams = self._hailo_imports["InputVStreamParams"]
        OutputVStreamParams = self._hailo_imports["OutputVStreamParams"]

        input_vstream_params = InputVStreamParams.make(self.network_group)
        output_vstream_params = OutputVStreamParams.make(self.network_group)

        # HEF expects uint8 NHWC input
        input_data = np.expand_dims(preprocessed, 0).astype(np.uint8)

        with InferVStreams(
            self.network_group, input_vstream_params, output_vstream_params
        ) as infer_pipeline:
            result = infer_pipeline.infer({self._input_name: input_data})

        # Extract output and normalize
        output_key = list(result.keys())[0]
        embedding = result[output_key].flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _infer_cpu(self, preprocessed):
        # ONNX expects NCHW
        input_tensor = np.expand_dims(preprocessed.transpose(2, 0, 1), 0).astype(
            np.float32
        )
        result = self.session.run(None, {self._input_name: input_tensor})
        embedding = result[0].flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def is_brownie(self, crop_rgb):
        """Return (is_brownie, similarity_score)."""
        if self.reference_embedding is None:
            return False, 0.0
        embedding = self.get_embedding(crop_rgb)
        similarity = float(np.dot(embedding, self.reference_embedding))
        return similarity >= self.similarity_threshold, similarity

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
