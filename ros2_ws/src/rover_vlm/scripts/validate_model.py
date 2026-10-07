#!/usr/bin/env python3
"""Model check before the runs (paper outline, section E), on the Jetson.

Scores a folder of observation images with the engine exactly like the node
(center square crop, two-prompt relevance), times the encoder, and - given human
ratings - reports the Spearman rank correlation between human and model.

    python3 validate_model.py --engine ../models/siglip2_base_patch16_224_fp16.engine \
        --meta ../models/siglip2_base_patch16_224.json --images ~/vlm_check --ratings ratings.csv

ratings.csv: "filename,rating" with a higher rating = more relevant for the search
(any scale; ties allowed). Without --ratings only the scores and latency are printed.
"""
import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rover_vlm.relevance import crop_columns, prompt_embedding, relevance  # noqa: E402
from rover_vlm.trt_image_encoder import load_meta, TrtImageEncoder  # noqa: E402


def ranks(x):
    """Average ranks (ties share the mean rank)."""
    x = np.asarray(x, float)
    order = x.argsort()
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--ratings')
    ap.add_argument('--similarity', choices=['scaled', 'cosine'], default='scaled')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--runs', type=int, default=50)
    args = ap.parse_args()

    meta = load_meta(args.meta)
    enc = TrtImageEncoder(args.engine, meta)
    pos, neg = prompt_embedding(meta['text_embeds']['positive']), prompt_embedding(meta['text_embeds']['negative'])
    scale = meta['logit_scale'] if args.similarity == 'scaled' else 1.0

    files = sorted(f for ext in ('jpg', 'jpeg', 'png') for f in glob.glob(os.path.join(args.images, f'*.{ext}')))
    if not files:
        raise SystemExit(f'no images in {args.images}')
    crops = []
    for f in files:
        img = cv2.imread(f)
        (u0, u1), = crop_columns(img.shape[1], img.shape[0], 1)
        crops.append(img[:, u0:u1] if img.shape[1] >= img.shape[0] else img)
    S = relevance(enc.embed(crops), pos, neg, scale, args.temperature)

    print(f'{meta["model_id"]}  similarity {args.similarity} (scale {scale:.2f}), temperature {args.temperature}')
    for f, s in sorted(zip(files, S), key=lambda x: -x[1]):
        print(f'  {s:.3f}  {os.path.basename(f)}')
    print(f'S range {S.min():.3f}-{S.max():.3f}, spread (std) {S.std():.3f}')

    for batch in sorted({1, enc.max_batch}):
        x = [crops[0]] * batch
        for _ in range(10):
            enc.embed(x)
        t = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            enc.embed(x)
            t.append((time.perf_counter() - t0) * 1000)
        print(f'latency batch {batch}: mean {np.mean(t):.1f} ms, p95 {np.percentile(t, 95):.1f} ms '
              f'(incl. preprocessing)')
    print(f'engine size {os.path.getsize(args.engine) / 1e6:.0f} MB')

    if args.ratings:
        with open(args.ratings) as f:
            human = {row[0].strip(): float(row[1]) for row in csv.reader(f) if row and not row[0].startswith('#')}
        pairs = [(human[os.path.basename(f)], s) for f, s in zip(files, S) if os.path.basename(f) in human]
        if len(pairs) < 3:
            raise SystemExit('need ratings for at least 3 of the images')
        h, m = zip(*pairs)
        print(f'Spearman rho (human vs model) over {len(pairs)} images: {spearman(h, m):.3f}')
    enc.close()


if __name__ == '__main__':
    main()
