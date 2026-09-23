"""Score coco/imgs/<concept>/*.jpg with the SAME pipeline the app uses:
  INT8 ONNX vision tower, 224x224 squash resize (bilinear), (x/255-0.5)/0.5, cosine vs concept prototypes.
Reports: top-1/top-3 accuracy per concept, matching vs non-matching cos distributions,
and a suggested heat mapping.
"""
import json, os, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IMGS = os.path.join(HERE, "coco", "imgs")
MODEL = os.path.join(HERE, "..", "seek-app", "assets", "model", "siglip2_vision_int8.onnx")
CONCEPTS = json.load(open(os.path.join(HERE, "..", "seek-app", "assets", "concepts.json"), encoding="utf-8"))


def preprocess(path):
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)  # squash, like the app
    a = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    return a.transpose(2, 0, 1)[None]


def main():
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    cids = [c["id"] for c in CONCEPTS["concepts"]]
    labels = {c["id"]: c["label"] for c in CONCEPTS["concepts"]}
    protos = np.array([c["prototype"] for c in CONCEPTS["concepts"]], dtype=np.float32)
    prompt_embeds = [np.array(c["prompt_embeds"], dtype=np.float32) for c in CONCEPTS["concepts"]]
    scale, bias = CONCEPTS["meta"]["logit_scale"], CONCEPTS["meta"]["logit_bias"]

    rows = []  # (true_cid, path, cos[20], maxp[20])
    for cid in sorted(os.listdir(IMGS)):
        d = os.path.join(IMGS, cid)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".jpg"):
                continue
            e = sess.run(None, {"pixel_values": preprocess(os.path.join(d, f))})[0][0]
            cos = protos @ e
            maxp = np.array([(pe @ e).max() for pe in prompt_embeds])
            rows.append((cid, f, cos, maxp))
    print(f"{len(rows)} images scored\n")

    # per-concept accuracy
    print(f"{'concept':12s} {'n':>2s} {'top1':>5s} {'top3':>5s} {'match_cos':>10s} {'other_cos':>10s} {'match_maxP':>10s}")
    all_match, all_other, all_match_p, all_other_p = [], [], [], []
    for k, cid in enumerate(cids):
        rs = [r for r in rows if r[0] == cid]
        if not rs:
            print(f"{cid:12s}  0   (no images)")
            continue
        top1 = np.mean([np.argmax(r[2]) == k for r in rs])
        top3 = np.mean([k in np.argsort(-r[2])[:3] for r in rs])
        m = [r[2][k] for r in rs]; mp = [r[3][k] for r in rs]
        o = [r[2][k] for r in rows if r[0] != cid]; op = [r[3][k] for r in rows if r[0] != cid]
        all_match += m; all_other += o; all_match_p += mp; all_other_p += op
        print(f"{cid:12s} {len(rs):2d} {top1:5.2f} {top3:5.2f} {np.mean(m):10.3f} {np.mean(o):10.3f} {np.mean(mp):10.3f}")

    am, ao = np.array(all_match), np.array(all_other)
    print(f"\nprototype cos: match mean {am.mean():.3f} (p10 {np.percentile(am,10):.3f}, p90 {np.percentile(am,90):.3f})"
          f" | non-match mean {ao.mean():.3f} (p90 {np.percentile(ao,90):.3f}, p99 {np.percentile(ao,99):.3f})")
    amp, aop = np.array(all_match_p), np.array(all_other_p)
    print(f"max-prompt cos: match mean {amp.mean():.3f} (p10 {np.percentile(amp,10):.3f})"
          f" | non-match mean {aop.mean():.3f} (p90 {np.percentile(aop,90):.3f}, p99 {np.percentile(aop,99):.3f})")
    # AUC-ish: fraction of (match, other) pairs where match > other
    auc = (am[:, None] > ao[None, :]).mean(); auc_p = (amp[:, None] > aop[None, :]).mean()
    print(f"pairwise AUC prototype={auc:.3f}  max-prompt={auc_p:.3f}")

    # heat mapping suggestion on logit = scale*cos + bias: put non-match p90 -> ~30, match p90 -> ~95
    lo_l = scale * np.percentile(ao, 90) + bias
    hi_l = scale * np.percentile(am, 90) + bias
    lo = lo_l - 0.3 * (hi_l - lo_l); hi = hi_l + 0.05 * (hi_l - lo_l)
    print(f"\ncurrent heat map logit_lo={CONCEPTS['meta']['heat']['logit_lo']} logit_hi={CONCEPTS['meta']['heat']['logit_hi']}")
    print(f"suggested       logit_lo={lo:.1f} logit_hi={hi:.1f}  (non-match p90 logit {lo_l:.1f}, match p90 logit {hi_l:.1f})")

    def heat(c, lo_, hi_):
        return float(np.clip((scale * c + bias - lo_) / (hi_ - lo_), 0, 1) * 99.9)
    print("\nheat with suggested map: match p10/p50/p90 = "
          + "/".join(f"{heat(np.percentile(am,q),lo,hi):.0f}" for q in (10, 50, 90))
          + " | non-match p50/p90/p99 = " + "/".join(f"{heat(np.percentile(ao,q),lo,hi):.0f}" for q in (50, 90, 99)))

    # ---- per-concept background calibration: bg = every image NOT in this concept's folder.
    # heat uses hybrid z = 0.5*z(prototype cos) + 0.5*z(max-prompt cos); pooled AUC 0.92 vs 0.88 raw on COCO set.
    zs_m, zs_o, hyb_m, hyb_o = [], [], [], []
    calib = {}
    for k, cid in enumerate(cids):
        bg = np.array([r[2][k] for r in rows if r[0] != cid]); mt = np.array([r[2][k] for r in rows if r[0] == cid])
        bgp = np.array([r[3][k] for r in rows if r[0] != cid]); mtp = np.array([r[3][k] for r in rows if r[0] == cid])
        mu, sd = float(bg.mean()), float(bg.std() + 1e-6); mup, sdp = float(bgp.mean()), float(bgp.std() + 1e-6)
        calib[cid] = {"bg_mean": round(mu, 5), "bg_std": round(sd, 5), "bgp_mean": round(mup, 5), "bgp_std": round(sdp, 5)}
        zs_m += list((mt - mu) / sd); zs_o += list((bg - mu) / sd)
        hyb_m += list(0.5 * (mt - mu) / sd + 0.5 * (mtp - mup) / sdp)
        hyb_o += list(0.5 * (bg - mu) / sd + 0.5 * (bgp - mup) / sdp)
    zm, zo = np.array(zs_m), np.array(zs_o); hm, ho = np.array(hyb_m), np.array(hyb_o)
    print(f"\npooled AUC: raw prototype={auc:.3f}  z-normalized={(zm[:,None]>zo[None,:]).mean():.3f}  z-hybrid(proto+maxP)={(hm[:,None]>ho[None,:]).mean():.3f}")
    print(f"hybrid z match p10/p50/p90 = {np.percentile(hm,10):.2f}/{np.percentile(hm,50):.2f}/{np.percentile(hm,90):.2f} | non-match p50/p90/p99 = {np.percentile(ho,50):.2f}/{np.percentile(ho,90):.2f}/{np.percentile(ho,99):.2f}")
    z_lo, z_hi = float(np.percentile(ho, 50)), float(np.percentile(hm, 90))
    def zheat(z): return float(np.clip((z - z_lo) / (z_hi - z_lo), 0, 1) * 99.9)
    print(f"hybrid heat map z_lo={z_lo:.2f} z_hi={z_hi:.2f}: match p10/p50/p90 = " + "/".join(f"{zheat(np.percentile(hm,q)):.0f}" for q in (10,50,90))
          + " | non-match p50/p90/p99 = " + "/".join(f"{zheat(np.percentile(ho,q)):.0f}" for q in (50,90,99))
          + f" | non-match >=80: {(np.array([zheat(z) for z in ho])>=80).mean()*100:.1f}%")
    if "--write" in sys.argv:
        cpath = os.path.join(HERE, "..", "seek-app", "assets", "concepts.json")
        for c in CONCEPTS["concepts"]:
            c["calibration"] = calib[c["id"]]
        CONCEPTS["meta"]["heat"] = {"mode": "zhybrid", "z_lo": round(z_lo, 3), "z_hi": round(z_hi, 3),
                                    "logit_lo": round(float(lo), 2), "logit_hi": round(float(hi), 2), "n_images": len(rows)}
        payload = json.dumps(CONCEPTS, ensure_ascii=False)  # serialize fully BEFORE truncating the file
        open(cpath, "w", encoding="utf-8").write(payload)
        print(f"wrote calibration for {len(calib)} concepts -> concepts.json ({len(payload)/1e6:.2f} MB)")

    # worst confusions
    print("\nworst confusions (true -> predicted):")
    conf = {}
    for r in rows:
        if r[0] in cids:
            p = cids[int(np.argmax(r[2]))]
            if p != r[0]:
                conf[(r[0], p)] = conf.get((r[0], p), 0) + 1
    for (t, p), n in sorted(conf.items(), key=lambda x: -x[1])[:10]:
        print(f"  {labels[t]}({t}) -> {labels[p]}({p}): {n}")


if __name__ == "__main__":
    main()
