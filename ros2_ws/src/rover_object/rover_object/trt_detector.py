"""TensorRT inference wrapper for the Ultralytics YOLO detector (person / fire).

GPU memory is handled through libcudart via ctypes so that no extra Python
packages (pycuda, cuda-python, torch) are required on the Jetson.
"""
import ctypes

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

    def free(self, ptr):
        self.lib.cudaFree(ptr)

    def stream(self):
        s = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(s)), 'cudaStreamCreate')
        return s

    def memcpy(self, dst, src, nbytes, kind, stream):
        self.check(self.lib.cudaMemcpyAsync(dst, src, nbytes, kind, stream), 'cudaMemcpyAsync')

    def sync(self, stream):
        self.check(self.lib.cudaStreamSynchronize(stream), 'cudaStreamSynchronize')


class TrtYoloDetector:
    def __init__(self, engine_path, conf_thres=0.4, iou_thres=0.5):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.cuda = _CudaRT()

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'Failed to deserialize engine: {engine_path}')
        self.context = self.engine.create_execution_context()
        self.stream = self.cuda.stream()

        self.input_name = None
        self.output_name = None
        self.host = {}
        self.device = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            self.host[name] = np.empty(shape, dtype=dtype)
            self.device[name] = self.cuda.malloc(self.host[name].nbytes)
            self.context.set_tensor_address(name, self.device[name].value)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        _, _, self.in_h, self.in_w = self.host[self.input_name].shape

    def close(self):
        for ptr in self.device.values():
            self.cuda.free(ptr)
        self.device.clear()
        if self.stream:
            self.cuda.lib.cudaStreamDestroy(self.stream)
            self.stream = None

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.in_h / h, self.in_w / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        pad_y, pad_x = (self.in_h - nh) / 2, (self.in_w - nw) / 2
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
        out = cv2.copyMakeBorder(
            resized, top, self.in_h - nh - top, left, self.in_w - nw - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return out, r, left, top

    def infer(self, bgr):
        """Return (boxes[N,4] xyxy in image pixels, scores[N], class_ids[N])."""
        lb, r, pad_x, pad_y = self._letterbox(bgr)
        blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        inp = self.host[self.input_name]
        inp[0] = blob.astype(inp.dtype) / 255.0
        inp = np.ascontiguousarray(inp)
        out = self.host[self.output_name]

        self.cuda.memcpy(self.device[self.input_name], inp.ctypes.data, inp.nbytes, _H2D, self.stream)
        if not self.context.execute_async_v3(self.stream.value):
            raise RuntimeError('TensorRT execution failed')
        self.cuda.memcpy(out.ctypes.data, self.device[self.output_name], out.nbytes, _D2H, self.stream)
        self.cuda.sync(self.stream)

        return self._postprocess(out[0], r, pad_x, pad_y, bgr.shape[:2])

    def _postprocess(self, pred, r, pad_x, pad_y, img_shape):
        # pred: [4 + num_classes, num_anchors] -> (cx, cy, w, h, cls scores...)
        pred = pred.T.astype(np.float32)
        cls_scores = pred[:, 4:]
        class_ids = cls_scores.argmax(axis=1)
        scores = cls_scores[np.arange(len(pred)), class_ids]
        keep = scores >= self.conf_thres
        if not np.any(keep):
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
        pred, scores, class_ids = pred[keep], scores[keep], class_ids[keep]

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / r
        ih, iw = img_shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, iw)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, ih)

        # Class-aware NMS: offset boxes per class so different classes never suppress each other.
        offset = class_ids[:, None].astype(np.float32) * 4096.0
        xywh = boxes + offset
        xywh[:, 2:] -= xywh[:, :2]
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.conf_thres, self.iou_thres)
        idx = np.array(idx, dtype=np.int64).reshape(-1)
        return boxes[idx], scores[idx], class_ids[idx]
