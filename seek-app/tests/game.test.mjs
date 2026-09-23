import assert from 'node:assert/strict';
import test from 'node:test';
import {
  addGuess, addShot, conceptIndex, decodeStore, emptyStore, isFinished, matchesGuess,
  MAX_SHOTS, newGame, nextPuzzleAt, puzzleDay, puzzleNumber, saveGame, shareMessage, stats,
} from '../src/game.ts';

const day = '2026-09-22';
const shot = { heat: 81.2, photo: '1-12345.jpg', latencyMs: 640 };

test('a shared Korean midnight switches date, number, and concept together', () => {
  const before = Date.parse('2026-09-22T14:59:59.999Z');
  const midnight = Date.parse('2026-09-22T15:00:00Z');
  assert.equal(puzzleDay(before), day);
  assert.equal(puzzleDay(midnight), '2026-09-23');
  assert.equal(nextPuzzleAt(before), midnight);
  assert.equal(nextPuzzleAt(midnight), midnight + 86_400_000);
  assert.equal(puzzleNumber(day), 1);
  assert.equal(conceptIndex(day, 20), 0);
  assert.equal(conceptIndex('2026-09-23', 20), 1);
  assert.equal(conceptIndex('2026-10-12', 20), 0);
});

test('twelve photos still allow guesses, but no thirteenth photo', () => {
  let game = newGame(day, 'summer');
  for (let n = 0; n < MAX_SHOTS; n++) game = addShot(game, shot);
  assert.equal(isFinished(game), false);
  assert.throws(() => addShot(game, shot));
  game = addGuess(game, '여름', ['여름']);
  assert.equal(game.solved, true);
  assert.throws(() => addGuess(game, '겨울', ['여름']));
  assert.throws(() => addShot(game, shot));
});

test('three wrong guesses end the game; empty and duplicate guesses do not consume a turn', () => {
  let game = newGame(day, 'summer');
  assert.throws(() => addGuess(game, '  ', ['여름']));
  game = addGuess(game, 'Winter', ['여름']);
  assert.throws(() => addGuess(game, ' winter ', ['여름']));
  assert.equal(game.guesses.length, 1);
  game = addGuess(game, '바다', ['여름']);
  game = addGuess(game, '여행', ['여름']);
  assert.equal(isFinished(game), true);
  assert.equal(game.solved, false);
  assert.throws(() => addShot(game, shot));
});

test('aliases normalize case, whitespace, fullwidth, and decomposed Hangul', () => {
  assert.equal(matchesGuess(' ＳＵＭＭＥＲ ', ['summer']), true);
  assert.equal(matchesGuess('여름'.normalize('NFD'), ['여름']), true);
  assert.equal(matchesGuess('따 뜻 함', ['따뜻함']), true);
  assert.equal(matchesGuess('', ['']), false);
});

test('restart keeps shot count, attempts, completion, onboarding, and events', () => {
  let game = addShot(newGame(day, 'summer'), shot);
  game = addGuess(game, '겨울', ['여름']);
  game = addGuess(game, '여름', ['여름']);
  const saved = saveGame({ ...emptyStore(), revision: 9, onboarded: true }, game, 'daily_complete');
  const restored = decodeStore(JSON.stringify(saved));
  assert.deepEqual(restored, saved);
  assert.equal(isFinished(restored.games[day]), true);
  assert.throws(() => addShot(restored.games[day], shot));
  assert.equal(newGame('2026-09-23', 'winter').shots.length, 0);
});

test('streak survives an unfinished today, breaks after a missed day, and counts each win once', () => {
  let saved = emptyStore();
  for (const d of ['2026-09-20', '2026-09-21']) {
    saved = saveGame(saved, addGuess(newGame(d, 'summer'), '여름', ['여름']));
  }
  assert.equal(stats(saved, day).streak, 2);
  assert.equal(stats(saved, '2026-09-23').streak, 0);
  const win = addGuess(newGame(day, 'summer'), '여름', ['여름']);
  saved = saveGame(saveGame(saved, win), win);
  assert.deepEqual(stats(saved, day), { streak: 3, wins: 3, played: 3 });
});

test('shared result has scores but never answer, guesses, photo, or local URI', () => {
  let game = addShot(newGame(day, 'summer'), shot);
  game = addGuess(game, '여름', ['여름']);
  const text = shareMessage(game);
  assert.match(text, /SEEK #001/);
  assert.match(text, /81\.2/);
  for (const privateValue of ['여름', 'summer', '12345', 'file://']) assert.equal(text.includes(privateValue), false);
  assert.throws(() => shareMessage(newGame(day, 'summer')));
});

test('corrupt or unsupported progress is rejected, not silently reset', () => {
  assert.throws(() => decodeStore('{'));
  assert.throws(() => decodeStore(JSON.stringify({ ...emptyStore(), version: 2 })));
  assert.throws(() => decodeStore(JSON.stringify({ ...emptyStore(), games: { [day]: { ...newGame(day, 'summer'), shots: [{ ...shot, heat: 200 }] } } })));
  assert.throws(() => decodeStore(JSON.stringify({ ...emptyStore(), games: { [day]: { ...newGame(day, 'summer'), shots: [{ ...shot, photo: '../../private.jpg' }] } } })));
  assert.throws(() => decodeStore(JSON.stringify({ ...emptyStore(), games: { '2026-02-30': newGame('2026-02-30', 'summer') } })));
  assert.throws(() => decodeStore(JSON.stringify({ ...emptyStore(), revision: -1 })));
});
