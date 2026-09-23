import { Directory, File, Paths } from 'expo-file-system';

import { decodeStore, emptyStore, GameStore } from './game';

const root = () => new Directory(Paths.document, 'seek');
const snapshot = (slot: number) => new File(root(), `progress-${slot}.json`);
const photoRoot = () => new Directory(Paths.cache, 'seek-photos');

export function readProgress(): { store: GameStore; recovered: boolean } {
  const candidates: GameStore[] = [];
  let damaged = 0;
  for (const slot of [0, 1]) {
    const file = snapshot(slot);
    if (!file.exists) continue;
    try { candidates.push(decodeStore(file.textSync())); } catch { damaged += 1; }
  }
  candidates.sort((a, b) => b.revision - a.revision);
  if (!candidates.length && damaged) throw new Error('저장된 기록을 읽지 못했어요. 앱을 다시 열어주세요.');
  return { store: candidates[0] ?? emptyStore(), recovered: damaged > 0 };
}

/** Alternating snapshots retain the previous revision if a write is interrupted. */
export function writeProgress(store: GameStore): GameStore {
  const next = { ...store, revision: store.revision + 1 };
  root().create({ idempotent: true, intermediates: true });
  const file = snapshot(next.revision % 2);
  file.write(JSON.stringify(next));
  // Do not consume a turn in the UI until the persisted revision can be read back.
  const saved = decodeStore(file.textSync());
  if (saved.revision !== next.revision) throw new Error('저장 공간을 확인한 뒤 다시 시도해주세요.');
  return next;
}

export function photoUri(day: string, name: string): string {
  return new File(photoRoot(), day, name).uri;
}

export async function keepPhoto(uri: string, day: string, shotNumber: number): Promise<string> {
  const folder = new Directory(photoRoot(), day);
  folder.create({ intermediates: true, idempotent: true });
  const name = `${shotNumber}-${Date.now()}.jpg`;
  await new File(uri).copy(new File(folder, name));
  return name;
}

export function discardPhoto(uri: string): void {
  try {
    // Only ever remove app-owned temporary files, never a gallery/user file.
    if (!uri.startsWith(Paths.cache.uri)) return;
    const file = new File(uri);
    if (file.exists) file.delete();
  } catch { /* Cache cleanup can be retried by the OS. */ }
}

export function cleanOldPhotos(today: string): void {
  const folder = photoRoot();
  if (!folder.exists) return;
  for (const entry of folder.list()) {
    if (entry instanceof Directory && entry.name !== today) entry.delete();
  }
}
