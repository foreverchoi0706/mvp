import { useCallback, useEffect, useRef, useState } from 'react';
import { AppState, Share } from 'react-native';
import type { CameraView } from 'expo-camera';

import {
  addGuess, addShot, GameStore, isFinished, MAX_SHOTS, newGame, nextPuzzleAt,
  puzzleDay, saveGame, shareMessage,
} from './game';
import { CONCEPTS, conceptForToday, embed, Engine, loadEngine, preprocess, scoreAgainst } from './scoring';
import { cleanOldPhotos, discardPhoto, keepPhoto, photoUri, readProgress, writeProgress } from './storage';
import { textGuard } from './textGuard';

// Share one native session across remounts; release it with the application process.
let enginePromise: Promise<Engine> | undefined;
function getEngine() {
  enginePromise ??= loadEngine().catch((error) => { enginePromise = undefined; throw error; });
  return enginePromise;
}

function initialProgress() {
  try { return { ...readProgress(), error: '' }; }
  catch { return { store: null, recovered: false, error: '저장된 기록을 열지 못했어요. 저장 공간을 확인한 뒤 다시 시도해주세요.' }; }
}

export function useDailyGame() {
  const [initial] = useState(initialProgress);
  const [store, setStore] = useState<GameStore | null>(initial.store);
  const storeRef = useRef<GameStore | null>(initial.store);
  const [today, setToday] = useState(puzzleDay);
  const [now, setNow] = useState(Date.now);
  const [active, setActive] = useState(AppState.currentState === 'active');
  const [loadError, setLoadError] = useState(initial.error);
  const [message, setMessage] = useState(initial.recovered ? '최근 저장에 문제가 있어 이전 기록을 복구했어요.' : '');
  const [engine, setEngine] = useState<Engine | null>(null);
  const [engineError, setEngineError] = useState(false);
  const [engineAttempt, setEngineAttempt] = useState(0);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const [sharing, setSharing] = useState(false);
  const shareRef = useRef(false);

  const hydrate = useCallback(() => {
    try {
      const result = readProgress();
      storeRef.current = result.store;
      setStore(result.store);
      setLoadError('');
      if (result.recovered) setMessage('최근 저장에 문제가 있어 이전 기록을 복구했어요.');
    } catch {
      setLoadError('저장된 기록을 열지 못했어요. 저장 공간을 확인한 뒤 다시 시도해주세요.');
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    getEngine().then((value) => { if (mounted) setEngine(value); }).catch(() => {
      if (mounted) setEngineError(true);
    });
    return () => { mounted = false; };
  }, [engineAttempt]);

  useEffect(() => {
    const tick = () => { setToday(puzzleDay()); setNow(Date.now()); };
    const timer = setInterval(tick, 30_000);
    const midnight = setTimeout(tick, Math.max(1, nextPuzzleAt() - Date.now() + 50));
    const subscription = AppState.addEventListener('change', (state) => {
      setActive(state === 'active');
      if (state === 'active') tick();
    });
    return () => { clearInterval(timer); clearTimeout(midnight); subscription.remove(); };
  }, [today]);

  useEffect(() => {
    try { cleanOldPhotos(today); } catch { /* Best-effort removal of expired cache photos. */ }
  }, [today]);

  function commit(next: GameStore) {
    const saved = writeProgress(next);
    storeRef.current = saved;
    setStore(saved);
  }

  function currentGame(day = puzzleDay()) {
    const current = storeRef.current;
    if (!current) throw new Error('기록을 불러오는 중이에요.');
    return current.games[day] ?? newGame(day, conceptForToday(day).id);
  }

  const game = store?.games[today] ?? newGame(today, conceptForToday(today).id);
  const concept = CONCEPTS.concepts.find((item) => item.id === game.conceptId);

  function start(): boolean {
    if (!storeRef.current || busyRef.current) return false;
    try {
      const current = currentGame();
      commit(saveGame({ ...storeRef.current, onboarded: true }, current,
        storeRef.current.games[current.day] ? undefined : 'daily_start'));
      setToday(current.day);
      setMessage('');
      return true;
    } catch { setMessage('기록을 저장하지 못했어요. 저장 공간을 확인해주세요.'); return false; }
  }

  async function shoot(camera: CameraView | null) {
    if (busyRef.current || !camera || !engine || !storeRef.current || !active) return;
    const current = currentGame();
    if (current.day !== today) { setToday(current.day); setMessage('새로운 문제가 열렸어요. 다시 촬영해주세요.'); return; }
    if (isFinished(current) || current.shots.length >= MAX_SHOTS) return;
    const target = CONCEPTS.concepts.find((item) => item.id === current.conceptId);
    if (!target) return;
    busyRef.current = true;
    setBusy(true);
    setMessage('');
    let capturedUri = '';
    let retainedUri = '';
    let committed = false;
    const started = Date.now();
    try {
      const photo = await camera.takePictureAsync({ quality: 0.6, skipProcessing: false });
      if (!photo?.uri) throw new Error('capture failed');
      capturedUri = photo.uri;
      const [guard, pre] = await Promise.allSettled([
        textGuard(photo.uri, photo.width, photo.height), preprocess(photo.uri),
      ]);
      if (guard.status === 'rejected') {
        setMessage('사진을 확인하지 못했어요. 촬영 기회는 그대로예요. 다시 시도해주세요.');
        return;
      }
      if (guard.value.blocked) {
        setMessage('글자가 많은 사진이에요. 주변 사물이나 풍경을 찍어주세요. 촬영 기회는 그대로예요.');
        return;
      }
      if (pre.status === 'rejected') throw pre.reason;
      const result = await embed(engine, pre.value.data);
      const score = scoreAgainst(result.vec, target);
      if (current.day !== puzzleDay()) {
        setToday(puzzleDay());
        setMessage('새로운 문제가 열렸어요. 오늘의 도전을 시작해주세요.');
        return;
      }
      const name = await keepPhoto(photo.uri, current.day, current.shots.length + 1);
      retainedUri = photoUri(current.day, name);
      // Recheck after asynchronous photo persistence as well as inference.
      if (current.day !== puzzleDay()) { setToday(puzzleDay()); return; }
      const next = addShot(currentGame(), { heat: score.heat, photo: name, latencyMs: Date.now() - started });
      commit(saveGame(storeRef.current, next, 'score_received'));
      committed = true;
    } catch {
      setMessage('촬영 결과를 저장하지 못했어요. 촬영 기회는 그대로예요. 다시 시도해주세요.');
    } finally {
      if (capturedUri) discardPhoto(capturedUri);
      if (retainedUri && !committed) discardPhoto(retainedUri);
      busyRef.current = false;
      setBusy(false);
    }
  }

  function guess(value: string): boolean {
    if (busyRef.current || !storeRef.current) return false;
    const current = currentGame();
    if (current.day !== today) { setToday(current.day); setMessage('문제가 바뀌었어요. 새로운 정답을 찾아보세요.'); return false; }
    const target = CONCEPTS.concepts.find((item) => item.id === current.conceptId);
    if (!target) return false;
    let next;
    try { next = addGuess(current, value, [target.label, ...target.aliases]); }
    catch (error) { setMessage(error instanceof Error ? error.message : '정답을 다시 입력해주세요.'); return false; }
    try {
      commit(saveGame(storeRef.current, next, next.solved ? 'daily_complete' : isFinished(next) ? 'daily_fail' : 'guess_wrong'));
      setMessage(next.solved || isFinished(next) ? '' : '아직 정답이 아니에요. 사진 속 공통점을 찾아보세요.');
      return true;
    } catch { setMessage('정답 시도를 저장하지 못했어요. 다시 시도해주세요.'); return false; }
  }

  async function share() {
    if (!storeRef.current || !isFinished(game) || shareRef.current) return;
    shareRef.current = true;
    setSharing(true);
    try {
      // This event means the share dialog was requested, not that a friend received it.
      commit(saveGame(storeRef.current, game, 'share_open'));
      await Share.share({ message: shareMessage(game), title: '오늘의 SEEK' });
    } catch { setMessage('공유 창을 열지 못했어요. 다시 시도해주세요.'); }
    finally { shareRef.current = false; setSharing(false); }
  }

  return {
    store, game, concept, today, now, active, busy, sharing, engineReady: !!engine, engineError,
    loadError, message, setMessage, hydrate, start, shoot, guess, share,
    retryEngine: () => { setEngineError(false); setEngineAttempt((value) => value + 1); },
  };
}
