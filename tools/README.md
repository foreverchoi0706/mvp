# tools

SigLIP2 → SEEK app assets. HF is blocked on the corp network, so the checkpoint comes from Google's GCS bucket.

```bash
UV_NATIVE_TLS=1 uv venv --python 3.12 .venv
UV_NATIVE_TLS=1 uv pip install --python .venv/bin/python torch transformers pillow numpy onnx onnxruntime onnxscript sentencepiece
curl -o ckpt/gemma_tokenizer.model https://storage.googleapis.com/big_vision/gemma_tokenizer.model
curl -o ckpt/siglip2_b16_224.npz  https://storage.googleapis.com/big_vision/siglip2/siglip2_b16_224.npz
.venv/bin/python convert_local.py   # npz -> hf/siglip2-base-patch16-224 (+ smoke test)
.venv/bin/python build_assets.py    # -> ../seek-app/assets/model/siglip2_vision_int8.onnx + concepts.json
.venv/bin/python quant_test.py      # optional: compare quantization variants
```

Concept list lives in `build_assets.py` (`CONCEPTS`). Edit, rerun, rebuild app.

## Score-quality check with real photos (COCO val2017)

```bash
curl -o coco/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -o coco/annotations_trainval2017.zip annotations/captions_val2017.json -d coco
.venv/bin/python pick_coco.py            # 6 images/concept by caption keyword + 40 neutral -> coco/imgs/
.venv/bin/python score_folder.py --write # top-1/top-3 per concept, AUC, per-concept z calibration -> concepts.json
```

`--write` stamps `calibration` (bg mean/std for prototype cos and max-prompt cos) into every concept and sets
`meta.heat = {mode: "zhybrid", z_lo, z_hi}`; the app maps hybrid z linearly to 0–99.9. Rerun after any prompt change.
Last run (160 imgs): hybrid AUC 0.934, match p50 64 / p90 100, non-match p90 37, false HOT (>=80) 1.1%.
