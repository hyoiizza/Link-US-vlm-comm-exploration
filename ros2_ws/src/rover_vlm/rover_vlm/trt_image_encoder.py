"""TensorRT runner for an exported VLM image encoder (see scripts/export_siglip2.py).

The engine maps pixel_values [B, 3, S, S] to L2-normalized image embeddings [B, D].
Preprocessing (size, mean, std) and the text side (prompt embeddings, logit scale)
come from the JSON written next to the ONNX file at export time, so the robot needs
neither PyTorch nor the tokenizer. GPU memory goes through libcudart via ctypes,
so no pycuda / cuda-python is needed on the Jetson.
"""
import ctypes
import json

import cv2
import numpy as np
import tensorrt as trt

_H2D = 1
_D2H = 2


class _CudaRT:
    def __init__(self):
        for name in ('libcudart.so', 'libcudart.so.12', '/usr/local/cuda/lib64/libcudart.so'):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise RuntimeError('libcudart not found')
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]

    def check(self, err, what):
        if err != 0:
            raise RuntimeError(f'{what} failed with CUDA error {err}')

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), 'cudaMalloc')
        return ptr

    def stream(self):
        s = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(s)), 'cudaStreamCreate')
        return s

    def memcpy(self, dst, src, nbytes, kind, stream):
        self.check(self.lib.cudaMemcpyAsync(dst, src, nbytes, kind, stream), 'cudaMemcpyAsync')

    def sync(self, stream):
        self.check(self.lib.cudaStreamSynchronize(stream), 'cudaStreamSynchronize')


def load_meta(path):
    """Export metadata: image_size, mean, std, embed_dim, logit_scale, prompts, text embeddings."""
    with open(path) as f:
        meta = json.load(f)
    for key in ('model_id', 'image_size', 'mean', 'std', 'logit_scale', 'text_embeds'):
        if key not in meta:
            raise ValueError(f'{path}: missing "{key}"')
    return meta


class TrtImageEncoder:
    def __init__(self, engine_path, meta):
        self.size = int(meta['image_size'])
        self.mean = np.array(meta['mean'], np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(meta['std'], np.float32).reshape(1, 3, 1, 1)
        self.cuda = _CudaRT()
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'Failed to deserialize engine: {engine_path}')
        self.context = self.engine.create_execution_context()
        self.stream = self.cuda.stream()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        # Largest batch of the optimization profile (a static engine reports its fixed shape).
        shape = tuple(self.engine.get_tensor_shape(self.input))
        self.max_batch = shape[0] if shape[0] > 0 else self.engine.get_tensor_profile_shape(self.input, 0)[2][0]
        self.dim = int(self.engine.get_tensor_shape(self.output)[-1])
        self.in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input))
        self.out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output))
        self.d_in = self.cuda.malloc(self.max_batch * 3 * self.size * self.size * np.dtype(self.in_dtype).itemsize)
        self.d_out = self.cuda.malloc(self.max_batch * self.dim * np.dtype(self.out_dtype).itemsize)
        self.context.set_tensor_address(self.input, self.d_in.value)
        self.context.set_tensor_address(self.output, self.d_out.value)

    def close(self):
        for ptr in (self.d_in, self.d_out):
            self.cuda.lib.cudaFree(ptr)
        if self.stream:
            self.cuda.lib.cudaStreamDestroy(self.stream)
            self.stream = None

    def preprocess(self, bgr_crops):
        rgb = [cv2.cvtColor(cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_LINEAR),
                            cv2.COLOR_BGR2RGB) for c in bgr_crops]
        x = np.stack(rgb).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        return np.ascontiguousarray(((x - self.mean) / self.std).astype(self.in_dtype))

    def embed(self, bgr_crops):
        """Unit image embeddings [len(crops), D] for BGR crops (any size, resized squarely)."""
        out = []
        for i in range(0, len(bgr_crops), self.max_batch):
            x = self.preprocess(bgr_crops[i:i + self.max_batch])
            n = x.shape[0]
            self.context.set_input_shape(self.input, x.shape)
            y = np.empty((n, self.dim), self.out_dtype)
            self.cuda.memcpy(self.d_in, x.ctypes.data, x.nbytes, _H2D, self.stream)
            if not self.context.execute_async_v3(self.stream.value):
                raise RuntimeError('TensorRT execution failed')
            self.cuda.memcpy(y.ctypes.data, self.d_out, y.nbytes, _D2H, self.stream)
            self.cuda.sync(self.stream)
            out.append(y.astype(np.float64))
        e = np.concatenate(out)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
