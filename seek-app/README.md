# SEEK — Daily Hunt MVP

주변 사진의 의미 점수를 단서로 매일 숨겨진 단어를 찾는 게임.
Camera → on-device SigLIP2 (ONNX Runtime) → Heat Score. No scoring server.

## Current product flow

- Korean onboarding and camera permission requested when the player starts.
- One shared puzzle per Korean calendar day; switches at 00:00 Asia/Seoul.
- 12 photos and 3 distinct answer attempts. After all photos are used, remaining guesses still work.
- Daily progress and successful-day streak persist across restarts. There is no production reset or answer preview.
- Rejected text photos, failed captures, and failed scoring do not consume a photo attempt.
- Results share through the native text share sheet, without answers, guesses, or photos.
- The 20 bundled concepts currently repeat every 20 days. This is an MVP content pool, not a full release schedule.

## Local data

`src/storage.ts` writes alternating versioned progress snapshots under the app document directory (`seek/progress-0.json`, `progress-1.json`). A damaged latest snapshot can fall back to the previous revision with a visible notice; if neither is readable, the app does not silently reset attempts.

Photos are kept only in the app cache for the current puzzle, and older puzzle photo folders are cleaned on launch/date change. The OS may evict cached images; scores remain available. Captured source images and preprocessing JPEGs are removed after use. No photo upload is implemented.

Up to 500 gameplay events are stored locally with event name, puzzle date, and timestamp. This is preparation for instrumentation, **not a connected analytics dashboard**. `share_open` measures requesting the OS dialog, not successful delivery. There is no invitation URL until a distribution URL is available.

## Checks

```bash
npm run typecheck
npm run lint
npm test
```

The Node tests cover midnight boundaries, attempt limits, answer normalization, persistence format, streaks, corrupt records, and spoiler-free sharing. Native camera and model execution require a development/native build, not Expo Go.

## Native development

```bash
npm install                      # runs patch-package (see patches/ for why)
# assets/model/siglip2_vision_int8.onnx + assets/concepts.json come from ../tools/build_assets.py
npx expo prebuild --platform android --no-install
cd android && ./gradlew assembleRelease
adb install -r app/build/outputs/apk/release/app-release.apk
```

- `src/scoring.ts` — model load (NNAPI → XNNPACK → CPU), 224×224 squash preprocess, cosine vs concept prototype, heat mapping.
- `src/textGuard.ts` — ML Kit text-area anti-cheat (paper with the answer written on it).
- `src/game.ts` — pure daily rules, streaks, persistence validation, and sharing format.
- `src/useDailyGame.ts` — inference, lifecycle/date changes, progress transactions, and sharing.
- `App.tsx` — one continuous Daily Hunt screen, onboarding, camera, history, guesses, and result card. No navigation stack is introduced.

Release distribution remains a separate step: no account system, remote analytics, leaderboard, ads, or purchases are wired up. The local daily schedule and answers are not tamper-resistant and must not back a prize-bearing ranking.

The Android build is configured for `arm64-v8a` through `expo-build-properties`. Set `ANDROID_HOME` to your local SDK when invoking Gradle directly. Add any required corporate-proxy truststore configuration in your local environment rather than committing machine-specific paths.
If you change anything under `node_modules/onnxruntime-react-native`, wipe `android/build/generated/autolinking` before building.

## Verification — 2026-09-22

- TypeScript and ESLint passed; 8 game-rule tests passed.
- Android release-variant APK compiled and installed on the connected arm64 emulator. The generated development signing key is not a production signing setup.
- Confirmed wrong-answer persistence across process restart/APK update, failure result screen after three guesses, and native share sheet without the answer or photos.
- The emulator reports zero available cameras. Camera failure preserves all 12 shots; a successful real capture → OCR → inference → photo persistence round trip still needs a camera-enabled device.
- iOS runtime behavior has not been exercised.
