#!/usr/bin/env python3
"""Make a SigLIP 2 image-encoder ONNX build and run well in TensorRT FP16 on the Jetson.

1. Fuse the decomposed LayerNorms (ReduceMean/Sub/Pow/Sqrt/Div, as exported at
   opset < 17) into LayerNormalization (opset 17). TensorRT then uses its
   normalization layer, which keeps the variance in FP32; the decomposed form
   overflowed in FP16 (TensorRT warns "Running layernorm after self-attention with
   FP16 Reduce or Pow may cause overflow"). Nothing else is fused, so every op stays
   a standard ONNX op TensorRT can parse.
2. Store the weights in FP16 (LayerNormalization and the graph inputs/outputs stay
   FP32). Halves the constants the builder has to place on the GPU; FP32 weights
   did not fit next to the desktop on the 8 GB Orin Nano.

    ~/vlm_tools/bin/python optimize_onnx.py <vision_encoder.onnx> [out.onnx]
Needs onnx, onnxconverter-common and onnxruntime (for its fusion tools).
"""
import sys

import onnx
from onnxconverter_common import float16
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_model


def main():
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace('.onnx', '_trt.onnx')
    options = FusionOptions('vit')
    for key in list(vars(options)):
        if key.startswith('enable_'):
            setattr(options, key, False)
    options.enable_layer_norm = True
    model = optimize_model(src, model_type='vit', num_heads=12, hidden_size=768,  # only used by the fusions left off
                           optimization_options=options, opt_level=0).model
    fused = sum(n.op_type == 'LayerNormalization' for n in model.graph.node)
    for o in model.opset_import:
        if o.domain in ('', 'ai.onnx'):
            o.version = max(o.version, 17)   # LayerNormalization is standard from opset 17
    model = float16.convert_float_to_float16(model, keep_io_types=True, op_block_list=['LayerNormalization'])
    onnx.save(model, dst)
    onnx.checker.check_model(dst)
    print(f'{dst}: {fused} LayerNormalization fused, FP16 weights')


if __name__ == '__main__':
    main()
