import { Asset } from 'expo-asset';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import jpeg from 'jpeg-js';
import { Platform } from 'react-native';
import { conceptIndex, matchesGuess, puzzleDay } from './game';
import { discardPhoto } from './storage';

import conceptsJson from '../assets/concepts.json';

export type Concept = {
  id: string;
  label: string;
  aliases: string[];
  difficulty: string;
  prompts: string[];
  prototype: number[];
  prompt_embeds: number[][];
  /** per-concept background stats from tools/score_folder.py --write (absent until calibrated) */
  calibration?: { bg_mean: number; bg_std: number; bgp_mean: number; bgp_std: number };
};

type ConceptsFile = {
  meta: {
    model: string;
    embed_dim: number;
    logit_scale: number;
    logit_bias: number;
    preprocess: { size: number; mean: number[]; std: number[]; resize_mode: string };
    heat: { mode?: 'zhybrid' | 'logit'; z_lo?: number; z_hi?: number; logit_lo: number; logit_hi: number };
  };
  concepts: Concept[];
};

export const CONCEPTS = conceptsJson as ConceptsFile;
const SIZE = CONCEPTS.meta.preprocess.size;

export type Engine = {
  session: InferenceSession;
  provider: string;
  inputName: string;
  outputName: string;
  loadMs: number;
};

export async function loadEngine(): Promise<Engine> {
  const t0 = Date.now();
  // Metro resolves bundled binary assets through a static require.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const asset = Asset.fromModule(require('../assets/model/siglip2_vision_int8.onnx'));
  await asset.downloadAsync();
  if (!asset.localUri) throw new Error('model asset has no localUri');
  const path = asset.localUri.replace(/^file:\/\//, '');

  // Try the platform accelerator first, fall back to default CPU.
  const attempts: string[][] = Platform.OS === 'android' ? [['nnapi'], ['xnnpack'], []] : [['coreml'], []];
  let lastErr: unknown;
  for (const eps of attempts) {
    try {
      const session = await InferenceSession.create(path, eps.length ? { executionProviders: eps as any } : {});
      return {
        session,
        provider: eps[0] ?? 'cpu',
        inputName: session.inputNames[0],
        outputName: session.outputNames[0],
        loadMs: Date.now() - t0,
      };
    } catch (e) {
      lastErr = e;
      console.warn('EP failed', eps, e);
    }
  }
  throw lastErr;
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = globalThis.atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** photo uri -> normalized CHW float tensor [1,3,SIZE,SIZE]. Squash resize, (x/255-0.5)/0.5. */
export async function preprocess(uri: string): Promise<{ data: Float32Array; ms: number }> {
  const t0 = Date.now();
  // ponytail: resize->jpeg->decode in JS. Fine for 224x224 (50k px). Go native if this ever dominates latency.
  const ref = await ImageManipulator.manipulate(uri).resize({ width: SIZE, height: SIZE }).renderAsync();
  const saved = await ref.saveAsync({ format: SaveFormat.JPEG, compress: 1, base64: true });
  ref.release();
  if (!saved.base64) throw new Error('no base64 from manipulator');
  let decoded: ReturnType<typeof jpeg.decode>;
  try {
    decoded = jpeg.decode(base64ToBytes(saved.base64), { useTArray: true, formatAsRGBA: true });
  } finally {
    discardPhoto(saved.uri);
  }
  if (decoded.width !== SIZE || decoded.height !== SIZE) {
    throw new Error(`unexpected decoded size ${decoded.width}x${decoded.height}`);
  }
  const px = decoded.data; // RGBA
  const plane = SIZE * SIZE;
  const data = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const o = i * 4;
    data[i] = px[o] / 127.5 - 1;
    data[plane + i] = px[o + 1] / 127.5 - 1;
    data[2 * plane + i] = px[o + 2] / 127.5 - 1;
  }
  return { data, ms: Date.now() - t0 };
}

export async function embed(engine: Engine, data: Float32Array): Promise<{ vec: Float32Array; ms: number }> {
  const t0 = Date.now();
  const input = new Tensor('float32', data, [1, 3, SIZE, SIZE]);
  const out = await engine.session.run({ [engine.inputName]: input });
  const vec = out[engine.outputName].data as Float32Array;
  return { vec, ms: Date.now() - t0 };
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export type Score = { cos: number; maxPromptCos: number; logit: number; z: number; heat: number };

export function scoreAgainst(vec: Float32Array, concept: Concept): Score {
  const { logit_scale, logit_bias, heat } = CONCEPTS.meta;
  const cos = dot(vec, concept.prototype);
  let maxPromptCos = -1;
  for (const p of concept.prompt_embeds) maxPromptCos = Math.max(maxPromptCos, dot(vec, p));
  const logit = logit_scale * cos + logit_bias;
  // Per-concept z-scores against a background image set (tools/score_folder.py --write). Raw cos means
  // different things per concept (bg mean ranges ~ -0.03..+0.02), so a global map mis-ranks concepts.
  // Hybrid of prototype-cos z and max-prompt-cos z: pooled AUC 0.93 vs 0.90 raw on the COCO check set.
  const cal = concept.calibration;
  const z = cal
    ? 0.5 * ((cos - cal.bg_mean) / cal.bg_std) + 0.5 * ((maxPromptCos - cal.bgp_mean) / cal.bgp_std)
    : NaN;
  let h: number;
  if (heat.mode === 'zhybrid' && cal && heat.z_lo !== undefined && heat.z_hi !== undefined) {
    h = (z - heat.z_lo) / (heat.z_hi - heat.z_lo);
  } else {
    // ponytail: fallback global linear map on SigLIP logit
    h = (logit - heat.logit_lo) / (heat.logit_hi - heat.logit_lo);
  }
  h = Math.min(1, Math.max(0, h)) * 99.9;
  return { cos, maxPromptCos, logit, z, heat: Math.round(h * 10) / 10 };
}

export function topConcepts(vec: Float32Array, k = 3): { concept: Concept; score: Score }[] {
  return CONCEPTS.concepts
    .map((c) => ({ concept: c, score: scoreAgainst(vec, c) }))
    .sort((a, b) => (isNaN(a.score.z) ? b.score.cos - a.score.cos : b.score.z - a.score.z))
    .slice(0, k);
}

export function heatLabel(h: number): string {
  if (h < 40) return 'COLD';
  if (h < 60) return 'WARMER';
  if (h < 80) return 'WARM';
  if (h < 90) return 'HOT';
  if (h < 97) return 'VERY HOT';
  return 'BOILING';
}

export function isCorrectGuess(guess: string, concept: Concept): boolean {
  return matchesGuess(guess, [concept.label, ...concept.aliases]);
}

/** Same puzzle for everyone on a given day. */
export function conceptForToday(day = puzzleDay()): Concept {
  return CONCEPTS.concepts[conceptIndex(day, CONCEPTS.concepts.length)];
}
