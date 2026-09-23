"""Compare INT8 quantization variants vs fp32 on real + synthetic images."""
import os, time, numpy as np, onnxruntime as ort
from PIL import Image
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import SiglipImageProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
FP32 = os.path.join(HERE, "out", "siglip2_vision_fp32.onnx")
ip = SiglipImageProcessor.from_pretrained(os.path.join(HERE, "hf", "siglip2-base-patch16-224"))
cats = Image.open(os.path.join(HERE, "ckpt", "coco_cats.jpg")).convert("RGB")
imgs = [cats, cats.rotate(90), cats.crop((0, 0, 300, 300)),
        Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8)),
        Image.new("RGB", (224, 224), (30, 120, 220)), Image.new("RGB", (224, 224), (255, 255, 255))]
names = ["cats", "cats_rot", "cats_crop", "noise", "blue", "white"]
px = ip(images=imgs, return_tensors="np")["pixel_values"].astype(np.float32)

def run(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    e = np.concatenate([s.run(None, {"pixel_values": px[i:i+1]})[0] for i in range(len(px))])
    s.run(None, {"pixel_values": px[:1]}); t0 = time.perf_counter()
    for _ in range(10): s.run(None, {"pixel_values": px[:1]})
    return e, (time.perf_counter() - t0) / 10 * 1000

ref, ms = run(FP32); print(f"fp32 {os.path.getsize(FP32)/1e6:.0f}MB {ms:.0f}ms")
pre = os.path.join(HERE, "out", "vision_fp32_pre.onnx")
if not os.path.exists(pre):
    try:
        quant_pre_process(FP32, pre, skip_symbolic_shape=True)
    except Exception as ex:
        print('pre_process failed, using raw fp32:', type(ex).__name__); pre = FP32
variants = {
    "int8_qint8_pertensor": dict(weight_type=QuantType.QInt8),
    "int8_qint8_perchannel": dict(weight_type=QuantType.QInt8, per_channel=True),
    "int8_quint8_perchannel_reduce": dict(weight_type=QuantType.QUInt8, per_channel=True, reduce_range=True),
    "int8_qint8_perchannel_matmul_only": dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"]),
}
for name, kw in variants.items():
    out = os.path.join(HERE, "out", f"vision_{name}.onnx")
    if not os.path.exists(out):
        quantize_dynamic(pre, out, **kw)
    e, ms = run(out)
    cos = (ref * e).sum(-1)
    print(f"{name:36s} {os.path.getsize(out)/1e6:5.1f}MB {ms:4.0f}ms  " + " ".join(f"{n}={c:.4f}" for n, c in zip(names, cos)))
