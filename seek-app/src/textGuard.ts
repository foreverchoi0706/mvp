/**
 * Anti-cheat: reject photos that are mostly text (someone writes "여름" on paper and shoots it).
 * CLIP-family models read text eagerly (typographic attack), so this runs BEFORE scoring.
 * On-device Google ML Kit, no network. Wired in once @react-native-ml-kit/text-recognition is installed.
 */
import TextRecognition, { TextRecognitionScript } from '@react-native-ml-kit/text-recognition';

export type TextGuardResult = {
  blocked: boolean;
  textAreaRatio: number; // sum of text block areas / image area
  blocks: number;
  chars: number;
  ms: number;
  sample: string;
};

// ponytail: single threshold. Tune from logs; consider max-block-height as a second signal if paper cheats slip through.
const AREA_RATIO_LIMIT = 0.12;
const CHARS_LIMIT = 40;

export async function textGuard(uri: string, width: number, height: number): Promise<TextGuardResult> {
  const t0 = Date.now();
  const res: any = await TextRecognition.recognize(uri, TextRecognitionScript.KOREAN);
  const blocks: any[] = res?.blocks ?? [];
  let area = 0;
  for (const b of blocks) {
    const f = b.frame ?? b.boundingBox;
    if (f) area += Math.max(0, f.width ?? 0) * Math.max(0, f.height ?? 0);
  }
  const imgArea = Math.max(1, width * height);
  const textAreaRatio = area / imgArea;
  const text: string = res?.text ?? '';
  const chars = text.replace(/\s/g, '').length;
  return {
    blocked: textAreaRatio > AREA_RATIO_LIMIT || chars > CHARS_LIMIT,
    textAreaRatio,
    blocks: blocks.length,
    chars,
    ms: Date.now() - t0,
    sample: text.slice(0, 40),
  };
}
