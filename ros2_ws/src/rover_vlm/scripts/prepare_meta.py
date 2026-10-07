#!/usr/bin/env python3
"""Robot metadata (the JSON the node reads) from a separately exported SigLIP 2 folder.

For exports that ship the encoders as ONNX next to the tokenizer instead of going
through export_siglip2.py:

    <model-dir>/vision_encoder.onnx    pixel_values -> image_embeds
    <model-dir>/text_encoder.onnx      input_ids [B, 64] -> text_embeds
    <model-dir>/tokenizer/tokenizer.json
    <model-dir>/preprocess_config.json image size, mean/std, logit_scale (log), logit_bias, text settings

Tokenizes the prompts, runs the text encoder once on the CPU (onnxruntime) and
writes image preprocessing, exp(logit_scale), prompts and their unit embeddings.
Needs `tokenizers` + `onnxruntime` (e.g. a venv with --system-site-packages):

    ~/vlm_tools/bin/python prepare_meta.py --model-dir ../siglip2-onnx/models/siglip2-base-patch16-224
"""
import argparse
import json
import math
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# Chosen 2026-10-01 on 1287 crops of run 1001_0151 (37 with a person, 59 blank walls; see
# ~/vlm_check/prompt_eval): person AUC 0.997, blank walls at the 8th percentile. The earlier
# pair ('an indoor area relevant / unlikely to be relevant for searching for a trapped person',
# cos 0.98) ranked people at the bottom (AUC 0.06): the negation is lost in the text encoder.
POSITIVE = ['a photo of a person', 'a photo of a human', 'a person lying on the floor',
            'a person sitting on the floor', 'an open doorway to a room',
            'a place where a person could be hiding']
NEGATIVE = ['a blank wall', 'a plain white wall', 'a close-up of a wall', 'a photo of a floor',
            'an empty corridor']


def token_ids(tokenizer, text, tp):
    """SigLIP 2 text input: lowercased, no <bos>, one <eos>, right-padded with <pad> to max_length.

    The tokenizer is used without its own padding / truncation (some exported
    tokenizer.json files already pad to 64, which would push <eos> behind the padding),
    and lowercased here because tokenizer.json does not carry do_lower_case.
    """
    ids = tokenizer.encode(text.lower()).ids
    if not tp.get('add_bos_token', False) and ids and ids[0] == tp['bos_token_id']:
        ids = ids[1:]
    if tp.get('add_eos_token', True) and (not ids or ids[-1] != tp['eos_token_id']):
        ids = ids + [tp['eos_token_id']]
    n = tp['max_length']
    if len(ids) > n:
        ids = ids[:n - 1] + [tp['eos_token_id']]
    return ids + [tp['pad_token_id']] * (n - len(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', help='default: rover_vlm/models/<model name>.json')
    ap.add_argument('--positive', nargs='+', default=POSITIVE)
    ap.add_argument('--negative', nargs='+', default=NEGATIVE)
    args = ap.parse_args()

    d = args.model_dir
    with open(os.path.join(d, 'preprocess_config.json')) as f:
        cfg = json.load(f)
    ip, tp = cfg['image_preprocessing'], cfg['text_preprocessing']
    if ip['image_size']['height'] != ip['image_size']['width']:
        raise SystemExit('square input expected')
    name = cfg['model_id'].split('/')[-1].replace('-', '_')
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', name + '.json')

    tokenizer = Tokenizer.from_file(os.path.join(d, 'tokenizer', 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    session = ort.InferenceSession(os.path.join(d, 'text_encoder.onnx'), providers=['CPUExecutionProvider'])

    def embed(prompts):
        ids = np.array([token_ids(tokenizer, p, tp) for p in prompts], dtype=np.int64)
        e = session.run(None, {'input_ids': ids})[0].astype(np.float64)
        return (e / np.linalg.norm(e, axis=1, keepdims=True)).tolist(), ids

    pos, pos_ids = embed(args.positive)
    neg, neg_ids = embed(args.negative)
    meta = {
        'model_id': cfg['model_id'],
        'image_size': ip['image_size']['height'],
        'mean': ip['normalize']['mean'],
        'std': ip['normalize']['std'],
        'embed_dim': cfg['embedding_dim'],
        # The checkpoint stores log(scale); SigLIP multiplies the cosine by exp(logit_scale).
        'logit_scale': math.exp(cfg['logit_scale']),
        'logit_bias': cfg['logit_bias'],
        'prompts': {'positive': args.positive, 'negative': args.negative},
        'prompt_token_ids': {'positive': [r[r > 0].tolist() for r in pos_ids],
                             'negative': [r[r > 0].tolist() for r in neg_ids]},
        'text_embeds': {'positive': pos, 'negative': neg},
        'source': os.path.abspath(d),
    }
    with open(out, 'w') as f:
        json.dump(meta, f, indent=1)
    cos = float(np.dot(np.mean(pos, 0), np.mean(neg, 0)))
    print(f'{out}\n{cfg["model_id"]}: {meta["image_size"]}px, dim {meta["embed_dim"]}, '
          f'scale exp({cfg["logit_scale"]:.3f}) = {meta["logit_scale"]:.1f}, bias {meta["logit_bias"]:.2f}\n'
          f'prompt tokens + {meta["prompt_token_ids"]["positive"]}\n'
          f'prompt tokens - {meta["prompt_token_ids"]["negative"]}\n'
          f'cos(positive, negative prompt) = {cos:.3f}')


if __name__ == '__main__':
    main()
