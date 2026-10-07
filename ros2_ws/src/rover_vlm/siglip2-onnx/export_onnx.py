import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

OUT_ROOT = os.path.dirname(os.path.abspath(__file__))

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

TEXT_MAX_LENGTH = 64
OPSET = 14


class VisionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model.get_image_features(pixel_values=pixel_values)
        if isinstance(out, torch.Tensor):
            return out
        return out.pooler_output


class TextWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        out = self.model.get_text_features(input_ids=input_ids)
        if isinstance(out, torch.Tensor):
            return out
        return out.pooler_output


def export_one(model_id: str, dir_name: str):
    print("=" * 80)
    print(f"Exporting {model_id}")
    print("=" * 80)

    out_dir = os.path.join(OUT_ROOT, "models", dir_name)
    os.makedirs(out_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    image_size = processor.image_processor.size["height"]
    assert processor.image_processor.size["height"] == processor.image_processor.size["width"]

    # ---- Vision encoder export ----
    vision_wrapper = VisionWrapper(model)
    dummy_pixel_values = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    vision_onnx_path = os.path.join(out_dir, "vision_encoder.onnx")
    torch.onnx.export(
        vision_wrapper,
        (dummy_pixel_values,),
        vision_onnx_path,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )
    print(f"Saved vision encoder -> {vision_onnx_path}")

    # ---- Text encoder export ----
    text_wrapper = TextWrapper(model)
    dummy_input_ids = torch.zeros(1, TEXT_MAX_LENGTH, dtype=torch.long)
    text_onnx_path = os.path.join(out_dir, "text_encoder.onnx")
    torch.onnx.export(
        text_wrapper,
        (dummy_input_ids,),
        text_onnx_path,
        input_names=["input_ids"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "text_embeds": {0: "batch_size"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )
    print(f"Saved text encoder -> {text_onnx_path}")

    # ---- Sanity check: compare PyTorch vs ONNXRuntime outputs ----
    import onnxruntime as ort
    import numpy as np

    texts = ["a photo of a cat", "a photo of two dogs playing"]
    tok = processor.tokenizer(
        texts, padding="max_length", truncation=True, max_length=TEXT_MAX_LENGTH, return_tensors="pt"
    )
    with torch.no_grad():
        torch_text_embeds = text_wrapper(tok["input_ids"]).numpy()

    sess = ort.InferenceSession(text_onnx_path, providers=["CPUExecutionProvider"])
    ort_text_embeds = sess.run(None, {"input_ids": tok["input_ids"].numpy().astype(np.int64)})[0]
    text_diff = np.abs(torch_text_embeds - ort_text_embeds).max()
    print(f"[check] text encoder max abs diff (torch vs onnxruntime): {text_diff:.6e}")

    from PIL import Image
    dummy_img = Image.new("RGB", (image_size, image_size), color=(120, 60, 200))
    img_inputs = processor.image_processor(images=[dummy_img], return_tensors="pt")
    with torch.no_grad():
        torch_img_embeds = vision_wrapper(img_inputs["pixel_values"]).numpy()

    sess_v = ort.InferenceSession(vision_onnx_path, providers=["CPUExecutionProvider"])
    ort_img_embeds = sess_v.run(None, {"pixel_values": img_inputs["pixel_values"].numpy().astype(np.float32)})[0]
    img_diff = np.abs(torch_img_embeds - ort_img_embeds).max()
    print(f"[check] vision encoder max abs diff (torch vs onnxruntime): {img_diff:.6e}")

    return {
        "model_id": model_id,
        "image_size": image_size,
        "patch_size": model.config.vision_config.patch_size,
        "projection_dim": model.config.text_config.projection_size,
        "vision_onnx_path": os.path.relpath(vision_onnx_path, OUT_ROOT),
        "text_onnx_path": os.path.relpath(text_onnx_path, OUT_ROOT),
        "text_max_diff": float(text_diff),
        "image_max_diff": float(img_diff),
    }


if __name__ == "__main__":
    results = []
    for m in MODELS:
        results.append(export_one(m["model_id"], m["dir_name"]))

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(json.dumps(results, indent=2))
