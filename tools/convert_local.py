"""Convert big_vision SigLIP2 npz (downloaded from GCS, HF is blocked here) -> HF SiglipModel dir.
Reuses key-mapping from transformers' convert_siglip_to_hf.py (fetched from GitHub).
Tokenizer is NOT saved as an HF tokenizer (needs gemma files from HF); we tokenize with
sentencepiece directly in build_assets.py.

Usage: .venv/bin/python convert_local.py
"""
import os, sys
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from convert_siglip_to_hf import (  # noqa: E402
    create_rename_keys, flatten_nested_dict, get_image_processor, get_siglip_config,
    read_in_q_k_v_head, rename_key, split_encoderblock_layers,
)
from transformers import SiglipModel  # noqa: E402
import sentencepiece as spm  # noqa: E402

MODEL_NAME = "siglip2-base-patch16-224"
NPZ = os.path.join(HERE, "ckpt", "siglip2_b16_224.npz")
SPM = os.path.join(HERE, "ckpt", "gemma_tokenizer.model")
OUT = os.path.join(HERE, "hf", MODEL_NAME)


def tokenize(sp, texts, max_len=64):
    # SigLIP2: lowercase, no BOS, append EOS(1), pad with 0 to max_len
    out = []
    for t in texts:
        ids = sp.encode(t.lower())[: max_len - 1] + [sp.eos_id()]
        ids += [sp.pad_id()] * (max_len - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)


@torch.no_grad()
def main():
    config = get_siglip_config(MODEL_NAME)
    print("loading npz ...")
    data = np.load(NPZ)
    state_dict = flatten_nested_dict(data)
    state_dict = split_encoderblock_layers(state_dict)
    for src, dest in create_rename_keys(config):
        rename_key(state_dict, src, dest, config)
    read_in_q_k_v_head(state_dict, config)

    model = SiglipModel(config).eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)
    assert not missing and not unexpected, "state dict mismatch"

    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    get_image_processor(MODEL_NAME).save_pretrained(OUT)
    print("saved ->", OUT)

    # smoke test: COCO cats image vs texts
    sp = spm.SentencePieceProcessor(model_file=SPM)
    img_path = os.path.join(HERE, "ckpt", "coco_cats.jpg")
    if os.path.exists(img_path):
        proc = get_image_processor(MODEL_NAME)
        px = proc(images=Image.open(img_path).convert("RGB"), return_tensors="pt")["pixel_values"]
        texts = ["two cats sleeping on a couch", "a dog running in a park", "a bowl of noodle soup", "an airplane"]
        out = model(input_ids=tokenize(sp, texts), pixel_values=px)
        probs = torch.sigmoid(out.logits_per_image)[0]
        for t, l, p in zip(texts, out.logits_per_image[0], probs):
            print(f"  {p.item():.3f}  logit {l.item():6.2f}  {t}")
        assert probs.argmax().item() == 0, "cats should win"
        print("smoke test OK")


if __name__ == "__main__":
    main()
