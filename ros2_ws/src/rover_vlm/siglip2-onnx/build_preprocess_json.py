import json
import os

from transformers import AutoModel, AutoProcessor

OUT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEXT_MAX_LENGTH = 64

RESAMPLE_NAMES = {
    0: "nearest",
    1: "lanczos",
    2: "bilinear",
    3: "bicubic",
    4: "box",
    5: "hamming",
}

MODELS = [
    {
        "model_id": "google/siglip2-base-patch16-224",
        "dir_name": "siglip2-base-patch16-224",
    },
    {
        "model_id": "google/siglip2-base-patch32-256",
        "dir_name": "siglip2-base-patch32-256",
    },
]


def build_one(model_id: str, dir_name: str) -> dict:
    out_dir = os.path.join(OUT_ROOT, "models", dir_name)
    os.makedirs(out_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    tok = processor.tokenizer
    img_proc = processor.image_processor

    # Save raw tokenizer artifacts (sentencepiece model / tokenizer.json / special tokens map)
    # so prompt embedding preprocessing can be reproduced without transformers installed.
    tokenizer_dir = os.path.join(out_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tok.save_pretrained(tokenizer_dir)

    image_size = img_proc.size["height"]
    assert img_proc.size["height"] == img_proc.size["width"]

    config = {
        "model_id": model_id,
        "architecture": model.config.model_type,
        "onnx": {
            "vision_encoder": f"models/{dir_name}/vision_encoder.onnx",
            "text_encoder": f"models/{dir_name}/text_encoder.onnx",
        },
        "embedding_dim": model.config.text_config.projection_size,
        "logit_scale": model.logit_scale.item(),
        "logit_bias": model.logit_bias.item(),
        "similarity_note": (
            "SigLIP uses a sigmoid (not softmax) similarity head: "
            "logits = logit_scale * (normalize(image_embeds) @ normalize(text_embeds).T) + logit_bias. "
            "L2-normalize both embeddings before comparing, and apply sigmoid to get a per-pair probability "
            "(no softmax across candidates is required, unlike CLIP)."
        ),
        "image_preprocessing": {
            "image_size": {"height": image_size, "width": image_size},
            "channel_order": "RGB",
            "layout": "NCHW",
            "resize": {
                "do_resize": img_proc.do_resize,
                "resample": RESAMPLE_NAMES.get(img_proc.resample, img_proc.resample),
                "resample_pil_code": img_proc.resample,
            },
            "rescale": {
                "do_rescale": img_proc.do_rescale,
                "rescale_factor": img_proc.rescale_factor,
            },
            "normalize": {
                "do_normalize": img_proc.do_normalize,
                "mean": img_proc.image_mean,
                "std": img_proc.image_std,
            },
            "steps_in_order": [
                "1. Convert image to RGB",
                f"2. Resize to {image_size}x{image_size} using bilinear interpolation (no aspect-ratio preservation, no center crop)",
                f"3. Rescale pixel values: pixel * {img_proc.rescale_factor} (i.e. divide by 255)",
                f"4. Normalize per channel: (pixel - mean) / std, mean={img_proc.image_mean}, std={img_proc.image_std}",
                "5. Transpose HWC -> CHW and add batch dimension -> shape (batch, 3, H, W), dtype float32",
            ],
        },
        "text_preprocessing": {
            "tokenizer_type": type(tok).__name__,
            "tokenizer_files_dir": f"models/{dir_name}/tokenizer/",
            "vocab_size": tok.vocab_size,
            "max_length": TEXT_MAX_LENGTH,
            "padding": "max_length",
            "truncation": True,
            "add_bos_token": tok.init_kwargs.get("add_bos_token", False),
            "add_eos_token": tok.init_kwargs.get("add_eos_token", True),
            "pad_token": tok.pad_token,
            "pad_token_id": tok.pad_token_id,
            "eos_token": tok.eos_token,
            "eos_token_id": tok.eos_token_id,
            "bos_token": tok.bos_token,
            "bos_token_id": tok.bos_token_id,
            "unk_token": tok.unk_token,
            "unk_token_id": tok.unk_token_id,
            "attention_mask_required": False,
            "canonicalization": "None required (raw text is tokenized as-is; no manual lowercasing/punctuation-stripping step).",
            "steps_in_order": [
                "1. Tokenize raw prompt text with the SentencePiece/Gemma tokenizer (no lowercasing needed)",
                "2. Do NOT prepend <bos> (add_bos_token=false)",
                "3. Append <eos> (id=1) after the tokenized text (add_eos_token=true)",
                f"4. Pad with <pad> (id=0) on the right up to max_length={TEXT_MAX_LENGTH}",
                f"5. If longer than {TEXT_MAX_LENGTH} tokens (including eos), truncate to {TEXT_MAX_LENGTH}",
                "6. Feed only input_ids (int64) to the ONNX text encoder; no attention_mask input is used",
            ],
        },
        "onnx_export": {
            "opset": 14,
            "vision_encoder": {
                "input": {"name": "pixel_values", "shape": ["batch_size", 3, image_size, image_size], "dtype": "float32"},
                "output": {"name": "image_embeds", "shape": ["batch_size", model.config.text_config.projection_size], "dtype": "float32"},
            },
            "text_encoder": {
                "input": {"name": "input_ids", "shape": ["batch_size", TEXT_MAX_LENGTH], "dtype": "int64"},
                "output": {"name": "text_embeds", "shape": ["batch_size", model.config.text_config.projection_size], "dtype": "float32"},
            },
        },
    }

    config_path = os.path.join(out_dir, "preprocess_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Wrote {config_path}")

    return config


if __name__ == "__main__":
    all_configs = {}
    for m in MODELS:
        cfg = build_one(m["model_id"], m["dir_name"])
        all_configs[m["dir_name"]] = cfg

    combined_path = os.path.join(OUT_ROOT, "siglip2_preprocess_config.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_configs, f, ensure_ascii=False, indent=2)
    print(f"Wrote combined config -> {combined_path}")
