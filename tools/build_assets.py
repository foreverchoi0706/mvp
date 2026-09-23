"""Build SEEK app assets from SigLIP2:
  1. export vision tower -> ONNX fp32 -> dynamic INT8
  2. embed concept prompts with text tower -> concepts.json
  3. sanity: fp32 vs int8 embedding agreement + desktop latency

Usage: .venv/bin/python build_assets.py [--model google/siglip2-base-patch16-224]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image
import sentencepiece as spm
from transformers import SiglipImageProcessor, SiglipModel

HERE = os.path.dirname(os.path.abspath(__file__))
APP_ASSETS = os.path.join(HERE, "..", "seek-app", "assets")
OUT_MODEL_DIR = os.path.join(APP_ASSETS, "model")
OUT_CONCEPTS = os.path.join(APP_ASSETS, "concepts.json")
LOCAL_HF = os.path.join(HERE, "hf", "siglip2-base-patch16-224")
SPM = os.path.join(HERE, "ckpt", "gemma_tokenizer.model")


def tokenize(sp, texts, max_len=64):
    """SigLIP2 text preproc: lowercase, no BOS, EOS=1, pad=0 to max_len."""
    out = []
    for t in texts:
        ids = sp.encode(t.lower())[: max_len - 1] + [sp.eos_id()]
        ids += [sp.pad_id()] * (max_len - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)

# EASY/NORMAL only for v1. HARD (자유·외로움·희망) excluded: CLIP-family models
# lose ~half their retrieval accuracy on abstract text (CVPR 2025 "Seeing the Abstract").
# Prompts are deliberately concrete descriptions, not the abstract noun alone.
CONCEPTS = [
    # id, ko label, aliases (accepted guesses), difficulty, english prompts
    ("summer", "여름", ["여름", "summer", "여름철"], "easy",
     ["a hot summer day", "summer vacation at the beach", "bright summer sunshine outdoors",
      "ice cream on a hot day", "people swimming in summer", "a fan blowing on a hot day"]),
    ("winter", "겨울", ["겨울", "winter", "겨울철"], "easy",
     ["a cold winter day", "snow covered street in winter", "a person wearing a thick coat and scarf",
      "a snowman", "frost on a window", "hot drink on a cold day"]),
    ("school", "학교", ["학교", "school"], "easy",
     ["a school classroom", "students at school", "a school desk and chalkboard",
      "school textbooks and a backpack", "a school building", "a school hallway with lockers"]),
    ("travel", "여행", ["여행", "travel", "trip"], "easy",
     ["going on a trip with a suitcase", "a passport and boarding pass", "an airplane at the airport",
      "a tourist taking photos", "a hotel room", "a map and travel bag"]),
    ("rain", "비", ["비", "rain", "장마"], "easy",
     ["rain falling on the street", "an umbrella in the rain", "raindrops on a window",
      "puddles on a rainy day", "a rainy grey sky", "wet pavement in the rain"]),
    ("morning", "아침", ["아침", "morning"], "easy",
     ["early morning sunrise", "breakfast on a table in the morning", "an alarm clock in the morning",
      "morning coffee and toast", "sunlight through curtains in the morning", "brushing teeth in the morning"]),
    ("exercise", "운동", ["운동", "exercise", "workout", "sports"], "easy",
     ["a person exercising at the gym", "people playing tennis", "a skateboarder doing a trick",
      "a person jogging outdoors", "a soccer game on a field", "athletes playing baseball", "dumbbells and a yoga mat"]),
    ("party", "파티", ["파티", "party"], "easy",
     ["a birthday party with balloons", "a party with cake and candles", "confetti and party decorations",
      "people celebrating at a party", "a gift box with a ribbon", "party hats and streamers"]),
    ("sea", "바다", ["바다", "sea", "ocean", "해변"], "easy",
     ["the ocean with waves", "a sandy beach by the sea", "blue sea water",
      "a seashell on the sand", "a boat on the sea", "the seaside horizon"]),
    ("rest", "휴식", ["휴식", "rest", "쉼", "relax"], "easy",
     ["a person relaxing on a sofa", "a cozy bed with pillows", "a cup of tea and a blanket",
      "a hammock for resting", "someone napping", "a quiet living room"]),
    ("speed", "속도", ["속도", "speed", "빠름"], "normal",
     ["a fast moving car with motion blur", "a speeding train", "a race car on a track",
      "a speedometer", "a sprinter running fast", "a motorcycle at high speed"]),
    ("health", "건강", ["건강", "health", "healthy"], "normal",
     ["a doctor with a stethoscope", "a person doing yoga in the morning", "fresh vegetables and fruit",
      "a glass of water and vitamins", "a person running for fitness", "a hospital room"]),
    ("study", "공부", ["공부", "study", "studying"], "normal",
     ["a student studying at a desk with books", "a person working on a laptop at a desk", "an open book with handwritten notes",
      "a library with bookshelves", "writing in a notebook with a pencil", "a stack of textbooks"]),
    ("adventure", "모험", ["모험", "adventure", "탐험"], "normal",
     ["a person hiking up a mountain with a backpack", "rock climbing on a cliff", "kayaking down a wild river",
      "camping in the wilderness", "mountain biking on a rugged trail", "exploring a cave with a flashlight"]),
    ("warmth", "따뜻함", ["따뜻함", "따뜻", "warm", "warmth"], "normal",
     ["a warm fireplace", "hands holding a warm mug", "a cozy wool blanket",
      "warm sunlight on skin", "a candle glowing", "a warm knitted sweater"]),
    ("order", "질서", ["질서", "order", "정리"], "normal",
     ["neatly organized shelves", "items arranged in a straight line", "a tidy desk with everything in place",
      "people standing in an orderly queue", "color sorted books on a shelf", "a grid of identical boxes"]),
    ("danger", "위험", ["위험", "danger", "dangerous"], "normal",
     ["a warning sign with a red triangle", "a cliff edge", "fire and smoke",
      "a caution tape barrier", "a sharp knife blade", "a high voltage sign"]),
    ("night", "밤", ["밤", "night", "야경"], "easy",
     ["a city at night with bright lights", "a dark night sky with stars", "the moon in the night sky",
      "a street lamp glowing at night", "a nighttime skyline", "a dark room lit by a lamp"]),
    ("nature", "자연", ["자연", "nature", "숲"], "easy",
     ["a forest with tall trees", "a green meadow with wildflowers", "a river flowing through a valley",
      "mountains and a lake", "leaves and plants in a garden", "a waterfall in the woods"]),
    ("food", "음식", ["음식", "food", "밥"], "easy",
     ["a plate of delicious food", "a bowl of rice and side dishes", "a table full of dishes",
      "a sandwich and fries", "a bowl of noodle soup", "fresh bread and pasta"]),
]


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LOCAL_HF)
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    torch.manual_seed(0)
    print("loading", args.model)
    model = SiglipModel.from_pretrained(args.model).eval()
    ip = SiglipImageProcessor.from_pretrained(args.model)
    sp = spm.SentencePieceProcessor(model_file=SPM)
    vision = model.vision_model
    logit_scale = float(model.logit_scale.exp().item())
    logit_bias = float(model.logit_bias.item())
    print(f"logit_scale={logit_scale:.3f} logit_bias={logit_bias:.3f}")
    print("image preprocessing:", ip.size, "mean", ip.image_mean, "std", ip.image_std,
          "resample", ip.resample, "do_center_crop", getattr(ip, "do_center_crop", None))

    # ---- 1. vision tower -> ONNX
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    fp32_path = os.path.join(HERE, "out", "siglip2_vision_fp32.onnx")  # not shipped
    int8_path = os.path.join(OUT_MODEL_DIR, "siglip2_vision_int8.onnx")
    dummy = torch.randn(1, 3, 224, 224)

    class VisionWrapper(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, pixel_values):
            out = self.v(pixel_values=pixel_values)
            emb = out.pooler_output
            return emb / emb.norm(dim=-1, keepdim=True)

    wrapper = VisionWrapper(vision).eval()
    if not args.skip_export:
        print("exporting fp32 onnx ...")
        with torch.no_grad():
            torch.onnx.export(
                wrapper, (dummy,), fp32_path,
                input_names=["pixel_values"], output_names=["image_embeds"],
                opset_version=17, dynamo=False, do_constant_folding=True,
            )
        print("quantizing int8 ...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        # per-channel, MatMul-only: fp32 cos agreement 0.998 on real photos vs 0.954 for default per-tensor
        # (LayerNorm/attention-softmax paths left in fp32; size +3MB, same speed). See quant_test.py.
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul"])
    for p in (fp32_path, int8_path):
        print(f"{os.path.basename(p)}: {os.path.getsize(p)/1e6:.1f} MB")

    # ---- 2. concept prototypes
    print("embedding concept prompts ...")
    concepts_out = []
    with torch.no_grad():
        for cid, ko, aliases, diff, prompts in CONCEPTS:
            tf = model.get_text_features(input_ids=tokenize(sp, prompts))
            tf = getattr(tf, 'pooler_output', tf)  # transformers>=5 returns ModelOutput
            t = tf.cpu().numpy()
            t = l2n(t)
            proto = l2n(t.mean(axis=0, keepdims=True))[0]
            concepts_out.append({
                "id": cid, "label": ko, "aliases": [a.lower() for a in aliases], "difficulty": diff,
                "prompts": prompts,
                "prototype": [round(float(v), 6) for v in proto],
                "prompt_embeds": [[round(float(v), 6) for v in row] for row in t],
            })
    meta = {
        "model": "google/siglip2-base-patch16-224 (converted from big_vision npz)", "embed_dim": int(len(concepts_out[0]["prototype"])),
        "logit_scale": logit_scale, "logit_bias": logit_bias,
        "preprocess": {"size": 224, "mean": ip.image_mean, "std": ip.image_std, "resize_mode": "squash"},
        # ponytail: global linear map on SigLIP logit; replace with per-concept percentile once play logs exist
        "heat": {"logit_lo": -14.0, "logit_hi": 4.0},
    }
    with open(OUT_CONCEPTS, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "concepts": concepts_out}, f, ensure_ascii=False)
    print(f"wrote {OUT_CONCEPTS} ({os.path.getsize(OUT_CONCEPTS)/1e6:.2f} MB), {len(concepts_out)} concepts")

    # ---- 3. sanity: fp32 vs int8 agreement, latency, cross-concept check on synthetic images
    import onnxruntime as ort
    s32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    imgs = [Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8)) for _ in range(3)]
    imgs.append(Image.new("RGB", (224, 224), (30, 120, 220)))   # blue block ~ sea/sky?
    imgs.append(Image.new("RGB", (224, 224), (255, 255, 255)))  # white ~ snow?
    px = ip(images=imgs, return_tensors="np")["pixel_values"].astype(np.float32)
    with torch.no_grad():
        ref = wrapper(torch.from_numpy(px)).numpy()
    # exported with static batch=1 (what the app uses) -> run one image at a time
    e32 = np.concatenate([s32.run(None, {"pixel_values": px[i:i+1]})[0] for i in range(len(px))])
    e8 = np.concatenate([s8.run(None, {"pixel_values": px[i:i+1]})[0] for i in range(len(px))])
    print("torch vs onnx fp32 cos:", np.round((ref * e32).sum(-1), 4))
    print("fp32 vs int8 cos      :", np.round((e32 * e8).sum(-1), 4))
    protos = np.array([c["prototype"] for c in concepts_out], dtype=np.float32)
    for name, e in (("blue", e8[3]), ("white", e8[4])):
        sims = protos @ e
        top = np.argsort(-sims)[:3]
        print(f"{name} block -> top3:", [(concepts_out[i]["label"], round(float(sims[i]), 3)) for i in top])
    one = px[:1]
    for label, sess in (("fp32", s32), ("int8", s8)):
        sess.run(None, {"pixel_values": one})
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            sess.run(None, {"pixel_values": one})
        print(f"desktop CPU latency {label}: {(time.perf_counter()-t0)/n*1000:.0f} ms/img")


if __name__ == "__main__":
    main()
