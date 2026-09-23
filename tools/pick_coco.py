"""Pick COCO val2017 images per SEEK concept by caption keywords and download them.
Output: coco/imgs/<concept_id>/<image_id>.jpg  (+ coco/imgs/_neutral/ random images)
"""
import json, os, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(HERE, "coco", "annotations", "captions_val2017.json")
OUT = os.path.join(HERE, "coco", "imgs")
PER_CONCEPT = 6
NEUTRAL = 40

# concept_id -> regex on captions (COCO is everyday scenes; some SEEK concepts are thin here)
KW = {
    "summer":      r"\b(beach|sunny|swimming|surf(ing|er)?|ice cream|popsicle)\b",
    "winter":      r"\b(snow|snowy|skiing|skier|snowboard|sled)\b",
    "school":      r"\b(classroom|school|students?|chalkboard|blackboard)\b",
    "travel":      r"\b(suitcase|luggage|airport|airplane|passport|tourists?)\b",
    "rain":        r"\b(rain|raining|rainy|umbrellas?|wet street)\b",
    "morning":     r"\b(breakfast|sunrise|alarm clock|cereal|morning)\b",
    "exercise":    r"\b(tennis|skateboard(ing|er)?|running|jogging|gym|soccer|baseball|frisbee|yoga)\b",
    "night":       r"\b(night|nighttime|at night|lit up|moon|dark sky)\b",
    "nature":      r"\b(forest|meadow|field of|river|lake|garden|wildflowers|woods)\b",
    "party":       r"\b(birthday|party|balloons?|candles?|celebrat\w+)\b",
    "sea":         r"\b(ocean|sea|waves|shore|beach|boats?|sailboat)\b",
    "rest":        r"\b(couch|sofa|sleeping|napping|laying on a bed|lying on a bed|relaxing)\b",
    "speed":       r"\b(racing|race|speeding|motorcycles?|fast|jet)\b",
    "health":      r"\b(vegetables?|salad|broccoli|fruit|healthy|carrots?)\b",
    "study":       r"\b(laptop|books?|desk|studying|notebook|library)\b",
    "adventure":   r"\b(hiking|mountain|climbing|kayak|canoe|trail|backpack)\b",
    "warmth":      r"\b(fireplace|fire place|candles?|blanket|cozy|hot chocolate)\b",
    "order":       r"\b(lined up|in a row|rows of|neatly|arranged|organized|stacked)\b",
    "danger":      r"\b(fire|flames|smoke|warning|caution|knife|knives|cliff)\b",
    "food":        r"\b(pizza|sandwich|plate of food|meal|dinner|lunch|soup|pasta|noodles)\b",
}


def main():
    random.seed(0)
    d = json.load(open(CAP))
    caps = {}
    for a in d["annotations"]:
        caps.setdefault(a["image_id"], []).append(a["caption"].lower())
    ids = list(caps)
    picked = {}
    used = set()
    for cid, pat in KW.items():
        rx = re.compile(pat)
        # require the keyword in >=2 of the 5 captions -> stronger label
        hits = [i for i in ids if sum(bool(rx.search(c)) for c in caps[i]) >= 2 and i not in used]
        random.shuffle(hits)
        picked[cid] = hits[:PER_CONCEPT]
        used.update(picked[cid])
        print(f"{cid:12s} candidates={len(hits):4d} picked={len(picked[cid])}")
    neutral = [i for i in ids if i not in used]
    random.shuffle(neutral)
    picked["_neutral"] = neutral[:NEUTRAL]

    jobs = []
    for cid, lst in picked.items():
        os.makedirs(os.path.join(OUT, cid), exist_ok=True)
        for i in lst:
            jobs.append((cid, i))
    meta = {"captions": {str(i): caps[i] for lst in picked.values() for i in lst}, "picked": {k: [str(i) for i in v] for k, v in picked.items()}}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), ensure_ascii=False)

    def dl(job):
        cid, i = job
        p = os.path.join(OUT, cid, f"{i:012d}.jpg")
        if os.path.exists(p):
            return 0
        urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{i:012d}.jpg", p)
        return 1
    with ThreadPoolExecutor(16) as ex:
        n = sum(ex.map(dl, jobs))
    print(f"downloaded {n} new / {len(jobs)} total -> {OUT}")


if __name__ == "__main__":
    main()
