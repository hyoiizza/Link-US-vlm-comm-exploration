import inspect
from transformers import AutoProcessor, AutoModel

MODEL_IDS = [
    "google/siglip2-base-patch16-224",
    "google/siglip2-base-patch32-256",
]

for model_id in MODEL_IDS:
    print("=" * 80)
    print(model_id)
    print("=" * 80)
    processor = AutoProcessor.from_pretrained(model_id)
    print("--- processor ---")
    print(type(processor))
    print("image_processor:", processor.image_processor)
    print("tokenizer type:", type(processor.tokenizer))
    print("tokenizer model_max_length:", processor.tokenizer.model_max_length)
    print("tokenizer padding_side:", processor.tokenizer.padding_side)
    print("tokenizer pad_token:", processor.tokenizer.pad_token, processor.tokenizer.pad_token_id)
    print("tokenizer eos_token:", processor.tokenizer.eos_token, processor.tokenizer.eos_token_id)
    print("tokenizer bos_token:", processor.tokenizer.bos_token, processor.tokenizer.bos_token_id)
    print("tokenizer unk_token:", processor.tokenizer.unk_token, processor.tokenizer.unk_token_id)
    print("tokenizer vocab_size:", processor.tokenizer.vocab_size)
    if hasattr(processor, "image_seq_length"):
        print("processor.image_seq_length:", processor.image_seq_length)

    model = AutoModel.from_pretrained(model_id)
    model.eval()
    print("model class:", type(model))
    print("--- config ---")
    print(model.config)

    print("--- get_text_features signature ---")
    print(inspect.signature(model.get_text_features))
    print("--- get_image_features signature ---")
    print(inspect.signature(model.get_image_features))

    # quick sanity run
    import torch
    texts = ["a photo of a cat", "a photo of a dog"]
    tok = processor.tokenizer(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    print("tokenizer output keys:", tok.keys())
    print("input_ids shape:", tok["input_ids"].shape)
    if "attention_mask" in tok:
        print("attention_mask unique values:", torch.unique(tok["attention_mask"]))
    with torch.no_grad():
        if "attention_mask" in tok:
            txt_feat = model.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        else:
            txt_feat = model.get_text_features(input_ids=tok["input_ids"])
    print("txt_feat type:", type(txt_feat))
    if isinstance(txt_feat, torch.Tensor):
        print("text feature shape:", txt_feat.shape)
    else:
        print("txt_feat attrs:", [a for a in dir(txt_feat) if not a.startswith("_")])
        for attr in ["pooler_output", "text_embeds", "last_hidden_state"]:
            val = getattr(txt_feat, attr, None)
            if val is not None:
                print(f"  .{attr} shape:", val.shape)

    import PIL.Image
    dummy_img = PIL.Image.new("RGB", (processor.image_processor.size["width"], processor.image_processor.size["height"]))
    img_inputs = processor.image_processor(images=[dummy_img], return_tensors="pt")
    print("image processor output keys:", img_inputs.keys())
    print("pixel_values shape:", img_inputs["pixel_values"].shape)
    with torch.no_grad():
        img_feat = model.get_image_features(pixel_values=img_inputs["pixel_values"])
    print("img_feat type:", type(img_feat))
    if isinstance(img_feat, torch.Tensor):
        print("image feature shape:", img_feat.shape)
    else:
        print("img_feat attrs:", [a for a in dir(img_feat) if not a.startswith("_")])
        for attr in ["pooler_output", "image_embeds", "last_hidden_state"]:
            val = getattr(img_feat, attr, None)
            if val is not None:
                print(f"  .{attr} shape:", val.shape)
    print()
