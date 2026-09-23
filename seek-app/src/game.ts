// Pure game rules: no camera, React Native, or filesystem dependencies.
export const MAX_SHOTS = 12;
export const MAX_GUESSES = 3;
export const DAY_MS = 86_400_000;
const KST_OFFSET = 9 * 60 * 60 * 1000;
const FIRST_DAY = Date.UTC(2026, 8, 22) / DAY_MS;

export type Shot = { heat: number; photo: string; latencyMs: number };
export type DailyGame = {
  day: string;
  conceptId: string;
  shots: Shot[];
  guesses: string[];
  solved: boolean;
};
export type LocalEvent = {
  name: 'daily_start' | 'score_received' | 'guess_wrong' | 'daily_complete' | 'daily_fail' | 'share_open';
  day: string;
  at: number;
};
export type GameStore = {
  version: 1;
  revision: number;
  onboarded: boolean;
  games: Record<string, DailyGame>;
  events: LocalEvent[];
};

/** Everyone uses the same Korean midnight, regardless of device timezone. */
export function puzzleDay(now = Date.now()): string {
  return new Date(now + KST_OFFSET).toISOString().slice(0, 10);
}

export function dayNumber(day: string): number {
  return Math.floor(Date.parse(`${day}T00:00:00Z`) / DAY_MS);
}

export function puzzleNumber(day: string): number {
  return dayNumber(day) - FIRST_DAY + 1;
}

export function conceptIndex(day: string, count: number): number {
  if (count < 1) throw new Error('No concepts available');
  const offset = puzzleNumber(day) - 1;
  return ((offset % count) + count) % count;
}

export function nextPuzzleAt(now = Date.now()): number {
  return (Math.floor((now + KST_OFFSET) / DAY_MS) + 1) * DAY_MS - KST_OFFSET;
}

export function emptyStore(): GameStore {
  return { version: 1, revision: 0, onboarded: false, games: {}, events: [] };
}

export function newGame(day: string, conceptId: string): DailyGame {
  return { day, conceptId, shots: [], guesses: [], solved: false };
}

export function isFinished(game: DailyGame): boolean {
  return game.solved || game.guesses.length >= MAX_GUESSES;
}

export function normalizeGuess(value: string): string {
  return value.normalize('NFKC').toLowerCase().replace(/\s+/gu, '');
}

export function matchesGuess(value: string, aliases: string[]): boolean {
  const normalized = normalizeGuess(value);
  return !!normalized && aliases.some((alias) => normalizeGuess(alias) === normalized);
}

export function addShot(game: DailyGame, shot: Shot): DailyGame {
  if (isFinished(game) || game.shots.length >= MAX_SHOTS) throw new Error('촬영 기회를 모두 사용했어요.');
  if (!Number.isFinite(shot.heat) || shot.heat < 0 || shot.heat > 99.9) throw new Error('점수를 다시 확인해주세요.');
  return { ...game, shots: [...game.shots, shot] };
}

export function addGuess(game: DailyGame, value: string, aliases: string[]): DailyGame {
  if (isFinished(game)) throw new Error('오늘의 도전은 끝났어요.');
  const guess = value.trim();
  if (!normalizeGuess(guess) || guess.length > 40) throw new Error('정답을 40자 이내로 입력해주세요.');
  if (game.guesses.some((previous) => normalizeGuess(previous) === normalizeGuess(guess))) {
    throw new Error('이미 시도한 단어예요. 다른 단어를 생각해보세요.');
  }
  return { ...game, guesses: [...game.guesses, guess], solved: matchesGuess(guess, aliases) };
}

export function saveGame(store: GameStore, game: DailyGame, name?: LocalEvent['name']): GameStore {
  return {
    ...store,
    games: { ...store.games, [game.day]: game },
    events: name ? [...store.events, { name, day: game.day, at: Date.now() }].slice(-500) : store.events,
  };
}

export function stats(store: GameStore, today: string) {
  const wins = Object.values(store.games).filter((game) => game.solved).map((game) => dayNumber(game.day));
  const won = new Set(wins);
  let cursor = dayNumber(today);
  if (!won.has(cursor)) cursor -= 1;
  let streak = 0;
  while (won.has(cursor)) { streak += 1; cursor -= 1; }
  return { streak, wins: wins.length, played: Object.values(store.games).filter(isFinished).length };
}

export function heatEmoji(heat: number): string {
  if (heat < 40) return '🧊';
  if (heat < 60) return '🌤';
  if (heat < 80) return '☀️';
  if (heat < 97) return '🔥';
  return '💥';
}

/** Deliberately excludes the answer, guesses, photos, and private local URIs. */
export function shareMessage(game: DailyGame): string {
  if (!isFinished(game)) throw new Error('도전을 마친 뒤 공유할 수 있어요.');
  const journey = game.shots.map((shot) => `${heatEmoji(shot.heat)} ${shot.heat.toFixed(1)}`).join('\n');
  return [
    `SEEK #${String(puzzleNumber(game.day)).padStart(3, '0')} · ${game.day}`,
    game.solved ? '오늘의 비밀 단어 발견! ✅' : '오늘은 아쉽게! 내일 다시 도전 🧩',
    journey || '사진 없이 추리했어요.',
    `${game.shots.length}/${MAX_SHOTS} shots · ${game.guesses.length}/${MAX_GUESSES} guesses`,
    '정답은 비밀. 정답은 화면 밖에 있다.',
  ].join('\n\n');
}

function record(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function validDay(value: unknown): value is string {
  return typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)
    && Number.isFinite(dayNumber(value)) && new Date(dayNumber(value) * DAY_MS).toISOString().slice(0, 10) === value;
}

/** Reject damaged snapshots rather than silently replenishing daily attempts. */
export function decodeStore(raw: string): GameStore {
  const value: unknown = JSON.parse(raw);
  if (!record(value) || value.version !== 1 || !Number.isSafeInteger(value.revision)
    || (value.revision as number) < 0 || typeof value.onboarded !== 'boolean' || !record(value.games)
    || !Array.isArray(value.events)) throw new Error('Invalid saved game');
  for (const [key, game] of Object.entries(value.games)) {
    if (!validDay(key) || !record(game) || game.day !== key || typeof game.conceptId !== 'string'
      || typeof game.solved !== 'boolean' || !Array.isArray(game.shots) || game.shots.length > MAX_SHOTS
      || !Array.isArray(game.guesses) || game.guesses.length > MAX_GUESSES
      || !game.guesses.every((guess) => typeof guess === 'string' && !!normalizeGuess(guess) && guess.length <= 40)
      || (game.solved && game.guesses.length === 0)
      || !game.shots.every((shot) => record(shot) && typeof shot.heat === 'number'
        && Number.isFinite(shot.heat) && shot.heat >= 0 && shot.heat <= 99.9
        && typeof shot.photo === 'string' && /^\d+-\d+\.jpg$/.test(shot.photo)
        && typeof shot.latencyMs === 'number' && Number.isFinite(shot.latencyMs) && shot.latencyMs >= 0)) {
      throw new Error('Invalid saved game');
    }
  }
  const names = ['daily_start', 'score_received', 'guess_wrong', 'daily_complete', 'daily_fail', 'share_open'];
  if (value.events.length > 500 || !value.events.every((event) => record(event)
    && names.includes(String(event.name)) && validDay(event.day)
    && typeof event.at === 'number' && Number.isFinite(event.at))) throw new Error('Invalid saved events');
  return value as GameStore;
}
