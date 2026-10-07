#!/usr/bin/env python3
"""Export a SigLIP 2 image encoder to ONNX, plus the prompt embeddings, for the robot.

Run on a machine with PyTorch + transformers (the robot has neither), then copy both
files to rover_vlm/models/ on each Jetson and build the engine there with
scripts/build_engine.sh.

    pip install torch transformers sentencepiece onnx
    python export_siglip2.py --model google/siglip2-base-patch16-224 --out-dir ../models

Writes
    <name>.onnx   pixel_values [B, 3, S, S] -> image_embeds [B, D] (L2-normalized),
                  dynamic batch (the node sends all crops of a keyframe at once)
    <name>.json   model id, image size, mean/std, logit scale/bias, prompts and their
                  L2-normalized text embeddings (the robot never runs the text encoder)

Fixed-resolution SigLIP 2 checkpoints only (…-patch16-224/256/384/512, …-patch32-256);
the NaFlex variants need a different preprocessing.
"""
import argparse
import json
import os

import torch
from transformers import AutoModel, AutoProcessor

# Paper outline, section B (initial prompt candidates).
POSITIVE = ['an indoor area relevant for searching for a trapped person']
NEGATIVE = ['an indoor area unlikely to be relevant for searching for a trapped person']


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model.get_image_features(pixel_values=pixel_values)
        e = out if torch.is_tensor(out) else out.pooler_output
        return e / e.norm(dim=-1, keepdim=True)


def text_embeds(model, processor, prompts):
    # SigLIP was trained with max_length padding; other padding changes the embeddings.
    inputs = processor(text=prompts, padding='max_length', max_length=64, return_tensors='pt')
    with torch.no_grad():
        out = model.get_text_features(input_ids=inputs['input_ids'])
    e = out if torch.is_tensor(out) else out.pooler_output
    return (e / e.norm(dim=-1, keepdim=True)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='google/siglip2-base-patch16-224')
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
    ap.add_argument('--positive', nargs='+', default=POSITIVE)
    ap.add_argument('--negative', nargs='+', default=NEGATIVE)
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()
    if 'naflex' in args.model:
        raise SystemExit('NaFlex checkpoints are not supported; use a fixed-resolution one.')

    name = args.model.split('/')[-1].replace('-', '_')
    os.makedirs(args.out_dir, exist_ok=True)
    onnx_path = os.path.join(args.out_dir, name + '.onnx')
    json_path = os.path.join(args.out_dir, name + '.json')

    # eager attention exports cleanly to ONNX (SDPA / flash kernels do not)
    model = AutoModel.from_pretrained(args.model, attn_implementation='eager').eval()
    processor = AutoProcessor.from_pretrained(args.model)
    ip = processor.image_processor
    size = ip.size['height'] if isinstance(ip.size, dict) else int(ip.size)

    meta = {
        'model_id': args.model,
        'image_size': size,
        'mean': list(ip.image_mean),
        'std': list(ip.image_std),
        'logit_scale': float(model.logit_scale.exp()),
        'logit_bias': float(model.logit_bias) if getattr(model, 'logit_bias', None) is not None else 0.0,
        'prompts': {'positive': args.positive, 'negative': args.negative},
        'text_embeds': {'positive': text_embeds(model, processor, args.positive),
                        'negative': text_embeds(model, processor, args.negative)},
    }

    encoder = ImageEncoder(model).eval()
    dummy = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        ref = encoder(dummy)
    meta['embed_dim'] = int(ref.shape[-1])
    kwargs = dict(input_names=['pixel_values'], output_names=['image_embeds'],
                  dynamic_axes={'pixel_values': {0: 'batch'}, 'image_embeds': {0: 'batch'}},
                  opset_version=args.opset, do_constant_folding=True)
    try:
        torch.onnx.export(encoder, (dummy,), onnx_path, dynamo=False, **kwargs)
    except TypeError:   # torch < 2.5 has no dynamo argument
        torch.onnx.export(encoder, (dummy,), onnx_path, **kwargs)

    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'{onnx_path}\n{json_path}\nimage {size}px, embed dim {meta["embed_dim"]}, '
          f'logit scale {meta["logit_scale"]:.2f}')

    try:   # optional check that the ONNX graph reproduces PyTorch
        import numpy as np
        import onnxruntime as ort
        x = torch.rand(3, 3, size, size) * 2 - 1
        got = ort.InferenceSession(onnx_path).run(None, {'pixel_values': x.numpy()})[0]
        with torch.no_grad():
            want = encoder(x).numpy()
        print(f'ONNX vs PyTorch max |diff| {np.abs(got - want).max():.2e}')
    except ImportError:
        print('onnxruntime not installed: skipped the ONNX check')


if __name__ == '__main__':
    main()
