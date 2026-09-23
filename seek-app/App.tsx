import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator, Image, Keyboard, KeyboardAvoidingView, Linking, Platform,
  Pressable, ScrollView, StyleSheet, Text, TextInput, View,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

import { isFinished, MAX_GUESSES, MAX_SHOTS, nextPuzzleAt, puzzleNumber, stats } from './src/game';
import { photoUri } from './src/storage';
import { useDailyGame } from './src/useDailyGame';

const colors = {
  paper: '#E8DFC9', ink: '#252722', muted: '#706B5D', line: '#BDB39C', white: '#F5EFDF',
  orange: '#9D382B', softOrange: '#EAD3BE', green: '#3F4B3C', softGreen: '#DCD4BF',
  desk: '#202522', brass: '#C6AD79',
};
const serif = Platform.select({ ios: 'Georgia', android: 'serif' });
const mono = Platform.select({ ios: 'Courier', android: 'monospace' });

function Action({ label, onPress, disabled = false, secondary = false }: {
  label: string; onPress: () => void; disabled?: boolean; secondary?: boolean;
}) {
  return (
    <Pressable accessibilityRole="button" accessibilityState={{ disabled }} onPress={onPress} disabled={disabled}
      style={({ pressed }) => [styles.action, secondary && styles.secondary, disabled && styles.disabled, pressed && styles.pressed]}>
      <Text style={[styles.actionText, secondary && styles.secondaryText]}>{label}</Text>
    </Pressable>
  );
}

function Photo({ uri, size = 74 }: { uri: string; size?: number }) {
  const [missing, setMissing] = useState(false);
  return missing ? (
    <View style={[styles.photo, styles.photoMissing, { width: size, height: size }]}>
      <Text style={styles.footnote}>사진 없음</Text>
    </View>
  ) : <Image source={{ uri }} style={[styles.photo, { width: size, height: size }]} onError={() => setMissing(true)} accessibilityLabel="내가 촬영한 사진" />;
}

function heatText(heat: number) {
  if (heat < 40) return '연관성이 희박한 증거';
  if (heat < 60) return '검토할 만한 단서';
  if (heat < 80) return '유력한 단서 발견';
  if (heat < 90) return '사건의 핵심에 접근';
  if (heat < 97) return '결정적 단서에 근접';
  return '매우 강한 연관성';
}

function DailyHunt() {
  const daily = useDailyGame();
  const [permission, requestPermission, refreshPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const inputRef = useRef<TextInput>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState(false);
  const [cameraKey, setCameraKey] = useState(0);
  const [permissionBusy, setPermissionBusy] = useState(false);
  const [guess, setGuess] = useState('');
  const [helpOpen, setHelpOpen] = useState(false);
  const { game, store } = daily;
  const finished = isFinished(game);
  const shotsLeft = MAX_SHOTS - game.shots.length;
  const guessesLeft = MAX_GUESSES - game.guesses.length;
  const latest = game.shots.at(-1);
  const best = game.shots.reduce((value, shot) => Math.max(value, shot.heat), 0);
  const records = store ? stats(store, daily.today) : { streak: 0, wins: 0, played: 0 };
  const remainingMinutes = Math.max(1, Math.ceil((nextPuzzleAt(daily.now) - daily.now) / 60_000));
  const nextLabel = `${Math.floor(remainingMinutes / 60)}시간 ${remainingMinutes % 60}분`;

  useEffect(() => {
    setCameraOpen(false);
    setCameraReady(false);
    setGuess('');
  }, [daily.today]);

  useEffect(() => {
    setCameraReady(false);
    if (daily.active) void refreshPermission().catch(() => {});
  }, [daily.active, refreshPermission]);

  useEffect(() => {
    if (finished) { setCameraOpen(false); Keyboard.dismiss(); }
  }, [finished]);

  async function openCamera() {
    if (permissionBusy || daily.busy) return;
    setPermissionBusy(true);
    try {
      if (permission && !permission.granted && !permission.canAskAgain) {
        await Linking.openSettings();
        return;
      }
      const result = permission?.granted ? permission : await requestPermission();
      if (!result.granted) {
        daily.setMessage('주변을 찍으려면 카메라 권한이 필요해요. 아래 버튼으로 다시 설정할 수 있어요.');
        return;
      }
      if (!daily.start()) return;
      setCameraError(false);
      setCameraReady(false);
      setCameraKey((value) => value + 1);
      setCameraOpen(true);
    } catch { daily.setMessage('카메라를 열지 못했어요. 다시 시도해주세요.'); }
    finally { setPermissionBusy(false); }
  }

  function submitGuess() {
    if (daily.guess(guess)) { setGuess(''); Keyboard.dismiss(); }
  }

  if (!store || daily.loadError || !daily.concept) {
    return (
      <SafeAreaView style={styles.root}>
        <StatusBar style="light" />
        <View style={styles.loading}>
          <Text style={styles.logo}>SEEK</Text>
          {!daily.loadError && store === null && <ActivityIndicator color={colors.orange} />}
          <Text style={styles.body}>{daily.loadError || (store ? '오늘의 문제를 불러오지 못했어요. 앱 업데이트를 확인해주세요.' : '오늘의 도전을 준비하고 있어요.')}</Text>
          {!!daily.loadError && <Action label="기록 다시 불러오기" onPress={daily.hydrate} />}
        </View>
      </SafeAreaView>
    );
  }

  const needsSettings = permission && !permission.granted && !permission.canAskAgain;
  const canShoot = cameraReady && daily.engineReady && !daily.busy && shotsLeft > 0 && daily.active;

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <StatusBar style="light" />
      <KeyboardAvoidingView style={styles.flex} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
          <View style={styles.header}>
            <View><Text style={styles.agency}>일상 미스터리 조사국</Text><Text style={styles.logo} accessibilityLabel="SEEK 탐정사무소">SEEK<Text style={styles.logoDot}> / 탐정사무소</Text></Text></View>
            <View style={styles.headerRight}>
              <Text style={styles.streak}>연속 해결  {String(records.streak).padStart(2, '0')}일</Text>
              <Pressable accessibilityRole="button" accessibilityLabel="수사 지침" hitSlop={12} onPress={() => setHelpOpen(!helpOpen)} style={styles.helpButton}>
                <Text style={styles.helpIcon}>수사 지침</Text>
              </Pressable>
            </View>
          </View>

          <View style={styles.folderTab}><Text style={styles.folderTabText}>사건 기록부</Text><Text style={styles.folderSerial}>SEEK / DAILY CASE</Text></View>
          <View style={styles.dossier}>
          <View style={styles.issueRow}>
            <Text style={styles.eyebrow}>CASE No. {String(puzzleNumber(daily.today)).padStart(3, '0')}</Text>
            <Text style={styles.date}>{daily.today.replaceAll('-', '.')}</Text>
          </View>

          <View style={styles.caseHeading}>
            <View style={styles.flex}><Text style={styles.fileCategory}>수사 의뢰서 · 단어 실종 사건</Text><Text style={styles.title}>{finished ? '사건 수사 보고서' : '사라진 단어를\n찾아주십시오.'}</Text></View>
            <View style={styles.caseStamp}><Text style={styles.caseStampText}>{finished ? (game.solved ? '해결' : '미결') : '미해결'}</Text><Text style={styles.stampSmall}>{finished && game.solved ? 'SOLVED' : 'UNSOLVED'}</Text></View>
          </View>
          <Text style={styles.subtitle}>{finished ? '현장 조사 종료. 오늘의 수사 기록을 봉합니다.' : '단서는 당신 주변에 있습니다. 사진 속 공통된 의미를 추리해 숨겨진 단어를 밝혀내세요.'}</Text>

          {(!store.onboarded || helpOpen) && (
            <View style={styles.instructions}>
              <Text style={styles.cardEyebrow}>조사국에서 전달한 수사 지침</Text>
              {[
                ['I', '증거를 수집할 것', '주변 사물과 풍경을 촬영하세요.'],
                ['II', '연관성을 살필 것', '숨겨진 단어에 가까울수록 점수가 높습니다.'],
                ['III', '최종 추리를 제출할 것', '증거 사진 12장. 정답 제출은 3회입니다.'],
              ].map(([number, title, description]) => (
                <View key={number} style={styles.instructionRow}>
                  <Text style={styles.step}>{number}</Text>
                  <View style={styles.flex}><Text style={styles.stepTitle}>{title}</Text><Text style={styles.small}>{description}</Text></View>
                </View>
              ))}
              <Text style={styles.privacy}>증거 사진은 이 기기에만 임시 보관됩니다. 다음 사건을 열면 삭제됩니다.</Text>
              {helpOpen && <Action label="지침 접기" secondary onPress={() => setHelpOpen(false)} />}
            </View>
          )}

          {!!daily.message && <View style={styles.notice} accessibilityLiveRegion="polite"><Text style={styles.noticeText}>{daily.message}</Text></View>}

          {finished ? (
            <View style={styles.resultCard}>
              <Text style={styles.cardEyebrow}>수사 대상의 정체</Text>
              <Text style={styles.answer}>{daily.concept.label}</Text>
              <View style={styles.resultBadge}><Text style={styles.resultBadgeText}>{game.solved ? '사건 해결 · 기록 완료' : '미결 사건 · 기록 보관'}</Text></View>
              <View style={styles.resultStats}>
                {[['최고 연관도', game.shots.length ? best.toFixed(1) : '—'], ['증거 사진', `${game.shots.length} / 12`], ['추리 제출', `${game.guesses.length} / 3`]].map(([label, value]) => (
                  <View key={label} style={styles.metric}><Text style={styles.metricValue}>{value}</Text><Text style={styles.small}>{label}</Text></View>
                ))}
              </View>
              <View style={styles.journey}><Text style={styles.cardEyebrow}>증거의 연관도 기록</Text><Text style={styles.journeyText}>{game.shots.map((shot) => shot.heat.toFixed(1)).join('  /  ') || '수집된 사진 증거 없음'}</Text></View>
              <Action label={daily.sharing ? '공유 창 여는 중…' : '수사 기록 공유하기  ↗'} onPress={() => void daily.share()} disabled={daily.sharing} />
              <Text style={styles.shareNote}>공유 사본에는 사진과 정답이 포함되지 않습니다.</Text>
              <View style={styles.nextPuzzle}><Text style={styles.small}>다음 사건 접수까지</Text><Text style={styles.nextTime}>{nextLabel}</Text><Text style={styles.footnote}>접수 시각  00:00 KST</Text></View>
            </View>
          ) : (
            <>
              <View style={styles.counterRow}>
                <View><Text style={styles.cardEyebrow}>01 / 현장 조사</Text><View style={styles.dots}>{Array.from({ length: MAX_SHOTS }, (_, i) => <View key={i} style={[styles.dot, i >= shotsLeft && styles.usedDot]} />)}</View></View>
                <Text style={styles.counter}><Text style={styles.counterStrong}>{shotsLeft}</Text> / {MAX_SHOTS}</Text>
              </View>

              {shotsLeft > 0 ? (
                <View style={styles.cameraCard}>
                  <View style={styles.filmLabel}><Text style={styles.filmLabelText}>증거 채집 장비</Text><Text style={styles.filmLabelText}>FRAME {String(game.shots.length + 1).padStart(2, '0')} / 12</Text></View>
                  {cameraOpen && permission?.granted && daily.active && !cameraError ? (
                    <View style={styles.viewfinder}>
                      <CameraView key={cameraKey} ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" animateShutter
                        onCameraReady={() => setCameraReady(true)} onMountError={() => { setCameraError(true); setCameraReady(false); }} />
                      <View pointerEvents="none" style={styles.viewfinderOverlay}>
                        <Text style={styles.cameraTag}>현장 기록 중</Text>
                        <View style={styles.target}><View style={styles.crosshairH} /><View style={styles.crosshairV} /></View>
                        <Text style={styles.cameraHint}>{daily.busy ? '증거를 분석하고 있습니다…' : '사물과 풍경에서 단서를 찾으세요'}</Text>
                      </View>
                    </View>
                  ) : (
                    <View style={styles.cameraPlaceholder}>
                      <View style={styles.evidenceArt} accessible={false}>
                        <View style={styles.evidenceSlip}><Text style={styles.slipCaption}>대상 불명</Text><View style={styles.redacted} /><View style={[styles.redacted, { width: 38 }]} /><Text style={styles.slipSerial}>EXHIBIT — ?</Text></View>
                        <View style={styles.lensOuter}><View style={styles.lensInner}><Text style={styles.lensQuestion}>?</Text></View><View style={styles.lensHandle} /></View>
                      </View>
                      <Text style={styles.cameraTitle}>{cameraError ? '장비 연결을 확인해주세요' : '현장에 단서가 남아 있습니다.'}</Text>
                      <Text style={styles.cameraSub}>{cameraError ? '다른 앱이 카메라를 사용 중인지 확인해보세요.' : '가장 가까운 사물부터 조사해보세요.'}</Text>
                    </View>
                  )}
                  <View style={styles.captureBar}>
                    {cameraOpen && !cameraError && permission?.granted ? (
                      <Action label={daily.busy ? '증거 분석 중…' : !daily.engineReady ? '조사 장비 준비 중…' : !cameraReady ? '카메라 준비 중…' : '증거 사진 촬영'} disabled={!canShoot} onPress={() => void daily.shoot(cameraRef.current)} />
                    ) : (
                      <Action label={permissionBusy ? '카메라 여는 중…' : needsSettings ? '설정에서 카메라 허용하기' : game.shots.length ? '현장 조사 계속하기' : '현장 조사 시작'} disabled={permissionBusy || daily.busy} onPress={() => void openCamera()} />
                    )}
                    {daily.engineError && <><Text style={styles.errorText}>점수 계산을 준비하지 못했어요.</Text><Action label="다시 준비하기" secondary onPress={daily.retryEngine} /></>}
                  </View>
                </View>
              ) : <View style={styles.instructions}><Text style={styles.stepTitle}>증거 수집을 마쳤습니다.</Text><Text style={styles.body}>사진 속 공통점을 바탕으로 추리를 제출하세요. 남은 제출 기회는 {guessesLeft}회입니다.</Text></View>}

              {!!latest && (
                <View style={styles.heatCard} accessibilityLiveRegion="polite">
                  <Photo uri={photoUri(game.day, latest.photo)} />
                  <View style={styles.flex}><Text style={styles.cardEyebrow}>증거 #{String(game.shots.length).padStart(2, '0')} · 분석 결과</Text><Text style={styles.heatDescription}>{heatText(latest.heat)}</Text><Text style={styles.small}>최고 연관도 {best.toFixed(1)}</Text></View>
                  <Text style={styles.heatNumber}>{latest.heat.toFixed(1)}</Text>
                </View>
              )}

              <View style={styles.guessCard}>
                <View style={styles.sectionHeading}><Text style={styles.sectionTitle}>02 / 추리 기록</Text><Text style={styles.guessesLeft}>제출 가능 {guessesLeft}회</Text></View>
                <View style={styles.inputRow}>
                  <TextInput ref={inputRef} accessibilityLabel="오늘의 비밀 단어" style={styles.input} value={guess} onChangeText={setGuess} placeholder="사건의 정체는…" placeholderTextColor="#817764" maxLength={40} autoCapitalize="none" autoCorrect={false} returnKeyType="done" editable={!daily.busy} onSubmitEditing={submitGuess} />
                  <Pressable accessibilityRole="button" accessibilityLabel="정답 제출" disabled={daily.busy || !guess.trim()} accessibilityState={{ disabled: daily.busy || !guess.trim() }} onPress={submitGuess} style={({ pressed }) => [styles.guessButton, (daily.busy || !guess.trim()) && styles.disabled, pressed && styles.pressed]}><Text style={styles.guessButtonText}>제출</Text></Pressable>
                </View>
                <Text style={styles.footnote}>연관도가 높은 증거들의 공통점을 찾아보세요.</Text>
                {game.guesses.length > 0 && <View style={styles.guessChips}>{game.guesses.map((value, i) => <View key={i} style={styles.guessChip}><Text style={styles.wrongGuess}>{value} ×</Text></View>)}</View>}
              </View>
            </>
          )}

          {game.shots.length > 0 && (
            <View style={styles.history}>
              <View style={styles.sectionHeading}><Text style={styles.sectionTitle}>03 / 증거 보관함</Text><Text style={styles.small}>{game.shots.length}건 접수</Text></View>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.photoStrip}>
                {game.shots.map((shot, i) => (
                  <View key={shot.photo} style={styles.historyItem}>
                    <Photo uri={photoUri(game.day, shot.photo)} size={88} />
                    <View style={styles.historyCaption}><Text style={styles.footnote}>{String(i + 1).padStart(2, '0')}</Text><Text style={styles.historyScore}>{shot.heat.toFixed(1)}</Text></View>
                  </View>
                ))}
              </ScrollView>
            </View>
          )}

          <View style={styles.footer}>
            <Text style={styles.footerText}>{records.played > 0 ? `수사 ${records.played}건  /  해결 ${records.wins}건` : '모든 일상에는 풀리지 않은 사건이 있다.'}</Text>
            <Text style={styles.footnote}>SEEK 탐정사무소  /  매일 자정, 새 사건 접수</Text>
          </View>
          </View>
          <Text style={styles.deskFooter}>관찰하라. 연결하라. 밝혀내라.</Text>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

export default function App() {
  return <SafeAreaProvider><DailyHunt /></SafeAreaProvider>;
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  root: { flex: 1, backgroundColor: colors.desk },
  content: { paddingHorizontal: 16, paddingBottom: 24, maxWidth: 620, width: '100%', alignSelf: 'center' },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingTop: 16, paddingBottom: 26, gap: 10 },
  agency: { color: colors.brass, fontSize: 9, letterSpacing: 2, marginBottom: 7 },
  logo: { fontFamily: serif, fontSize: 31, fontWeight: '700', letterSpacing: 1, color: colors.white },
  logoDot: { fontFamily: undefined, fontSize: 11, fontWeight: '400', letterSpacing: 0, color: '#C0BEAF' },
  headerRight: { alignItems: 'flex-end', gap: 7 },
  streak: { fontFamily: mono, fontSize: 10, color: '#C0BEAF' },
  helpButton: { minHeight: 32, justifyContent: 'center', borderBottomWidth: 1, borderColor: '#777B6B' },
  helpIcon: { fontSize: 11, color: colors.white },
  folderTab: { alignSelf: 'flex-start', backgroundColor: '#C6B58F', borderTopLeftRadius: 3, borderTopRightRadius: 16, paddingHorizontal: 18, paddingTop: 10, paddingBottom: 9, flexDirection: 'row', gap: 28, alignItems: 'center' },
  folderTabText: { color: colors.ink, fontSize: 12, fontWeight: '700' },
  folderSerial: { color: '#5F5745', fontFamily: mono, fontSize: 8 },
  dossier: { backgroundColor: colors.paper, paddingHorizontal: 20, paddingTop: 20, paddingBottom: 20, gap: 20, borderTopWidth: 4, borderTopColor: '#C6B58F', borderBottomWidth: 5, borderBottomColor: '#B6A581' },
  issueRow: { flexDirection: 'row', justifyContent: 'space-between', borderBottomWidth: 1, borderColor: colors.ink, paddingBottom: 10, gap: 12 },
  eyebrow: { fontFamily: mono, fontSize: 11, letterSpacing: 1.2, fontWeight: '700', color: colors.ink },
  date: { fontFamily: mono, fontSize: 10, color: colors.muted },
  caseHeading: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  fileCategory: { color: colors.muted, fontSize: 10, marginBottom: 12, letterSpacing: 0.5 },
  title: { fontFamily: serif, fontSize: 28, lineHeight: 40, letterSpacing: -1.5, fontWeight: '700', color: colors.ink },
  caseStamp: { borderWidth: 2, borderColor: colors.orange, padding: 7, alignItems: 'center', transform: [{ rotate: '-9deg' }] },
  caseStampText: { fontSize: 19, fontWeight: '800', color: colors.orange, letterSpacing: 2 },
  stampSmall: { fontFamily: mono, fontSize: 7, letterSpacing: 1, color: colors.orange, marginTop: 3 },
  subtitle: { fontSize: 12, lineHeight: 21, color: '#514E43', marginTop: -4 },
  instructions: { paddingVertical: 16, borderTopWidth: 1, borderBottomWidth: 1, borderColor: colors.line, gap: 12 },
  cardEyebrow: { fontSize: 10, fontWeight: '700', color: colors.muted, letterSpacing: 0.5 },
  instructionRow: { flexDirection: 'row', gap: 12, alignItems: 'flex-start' },
  step: { fontFamily: serif, color: colors.orange, fontSize: 13, width: 22, paddingTop: 1 },
  stepTitle: { color: colors.ink, fontSize: 12, fontWeight: '700', marginBottom: 3 },
  small: { color: colors.muted, fontSize: 11, lineHeight: 18 },
  body: { color: colors.muted, fontSize: 13, lineHeight: 22 },
  privacy: { color: colors.muted, fontSize: 10, lineHeight: 17, borderTopWidth: 1, borderColor: colors.line, paddingTop: 10 },
  notice: { backgroundColor: colors.softOrange, borderLeftWidth: 3, borderLeftColor: colors.orange, padding: 12 },
  noticeText: { color: '#713123', fontSize: 12, lineHeight: 20 },
  counterRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12 },
  dots: { flexDirection: 'row', gap: 4, marginTop: 10 },
  dot: { width: 11, height: 15, borderWidth: 1, borderColor: colors.ink, backgroundColor: colors.ink },
  usedDot: { backgroundColor: 'transparent', borderColor: colors.line },
  counter: { fontFamily: mono, color: colors.muted, fontSize: 12 },
  counterStrong: { color: colors.ink, fontSize: 22, fontWeight: '700' },
  cameraCard: { backgroundColor: colors.desk, padding: 7 },
  filmLabel: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 5, paddingVertical: 8, gap: 8 },
  filmLabelText: { color: colors.brass, fontFamily: mono, fontSize: 8, letterSpacing: 1 },
  viewfinder: { height: 230, backgroundColor: colors.ink },
  viewfinderOverlay: { flex: 1, padding: 12, justifyContent: 'space-between', alignItems: 'center' },
  cameraTag: { backgroundColor: '#202522CC', color: colors.white, fontSize: 10, paddingHorizontal: 10, paddingVertical: 5 },
  cameraHint: { backgroundColor: '#202522CC', color: colors.white, fontSize: 10, padding: 7 },
  target: { width: 120, height: 90, borderWidth: 1, borderColor: '#E8DFC999', justifyContent: 'center', alignItems: 'center' },
  crosshairH: { width: 20, height: 1, backgroundColor: colors.paper },
  crosshairV: { height: 20, width: 1, backgroundColor: colors.paper, position: 'absolute' },
  cameraPlaceholder: { minHeight: 190, backgroundColor: '#303730', borderWidth: 1, borderColor: '#53594B', padding: 17, alignItems: 'center', justifyContent: 'center', gap: 8 },
  evidenceArt: { height: 82, width: 132, marginBottom: 5 },
  evidenceSlip: { position: 'absolute', left: 8, top: 1, width: 73, height: 73, padding: 9, backgroundColor: '#CBBE9E', transform: [{ rotate: '-11deg' }] },
  slipCaption: { fontSize: 8, color: '#443E31', marginBottom: 6 },
  redacted: { height: 5, width: 49, backgroundColor: '#55503D', marginBottom: 4 },
  slipSerial: { fontFamily: mono, fontSize: 6, color: '#443E31', marginTop: 5 },
  lensOuter: { position: 'absolute', left: 57, top: 8, width: 54, height: 54, borderRadius: 27, borderWidth: 3, borderColor: '#BBA375', justifyContent: 'center', alignItems: 'center', backgroundColor: '#303730' },
  lensInner: { width: 42, height: 42, borderRadius: 21, borderWidth: 1, borderColor: '#6B725E', alignItems: 'center', justifyContent: 'center' },
  lensQuestion: { fontFamily: serif, color: '#D9CAAC', fontSize: 28 },
  lensHandle: { position: 'absolute', width: 8, height: 29, backgroundColor: '#BBA375', right: -9, bottom: -22, transform: [{ rotate: '-39deg' }] },
  cameraTitle: { fontFamily: serif, color: colors.white, fontSize: 16, textAlign: 'center' },
  cameraSub: { color: '#B3B8A6', fontSize: 10, textAlign: 'center', lineHeight: 18 },
  captureBar: { paddingTop: 7, gap: 8 },
  action: { minHeight: 49, paddingHorizontal: 14, paddingVertical: 14, backgroundColor: colors.orange, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: '#7D2B20' },
  actionText: { color: colors.white, fontWeight: '700', fontSize: 13, textAlign: 'center', letterSpacing: 1 },
  secondary: { backgroundColor: 'transparent', borderColor: colors.line },
  secondaryText: { color: colors.ink },
  disabled: { opacity: 0.45 },
  pressed: { opacity: 0.75 },
  errorText: { color: '#E9C7A2', fontSize: 11, textAlign: 'center' },
  heatCard: { paddingVertical: 14, flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: 10, borderTopWidth: 1, borderBottomWidth: 1, borderColor: colors.line },
  photo: { backgroundColor: colors.softGreen, borderWidth: 4, borderColor: colors.white },
  photoMissing: { alignItems: 'center', justifyContent: 'center' },
  heatDescription: { fontSize: 12, fontWeight: '700', color: colors.ink, marginVertical: 5 },
  heatNumber: { fontFamily: mono, fontSize: 26, fontWeight: '700', color: colors.orange, letterSpacing: -1 },
  guessCard: { gap: 12, paddingTop: 4 },
  sectionHeading: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' },
  sectionTitle: { fontSize: 12, fontWeight: '700', color: colors.ink, letterSpacing: 1 },
  guessesLeft: { color: colors.orange, fontSize: 10 },
  inputRow: { flexDirection: 'row', gap: 0, borderBottomWidth: 1, borderColor: colors.ink },
  input: { flex: 1, minHeight: 51, paddingHorizontal: 10, fontSize: 15, color: colors.ink, backgroundColor: '#EFE7D5' },
  guessButton: { backgroundColor: colors.ink, paddingHorizontal: 19, alignItems: 'center', justifyContent: 'center' },
  guessButtonText: { color: colors.white, fontSize: 12, fontWeight: '700' },
  footnote: { fontSize: 9, lineHeight: 16, color: colors.muted },
  guessChips: { gap: 0 },
  guessChip: { paddingVertical: 8, borderBottomWidth: 1, borderColor: colors.line },
  wrongGuess: { fontSize: 12, color: colors.muted, textDecorationLine: 'line-through' },
  history: { gap: 14, borderTopWidth: 1, borderColor: colors.ink, paddingTop: 17 },
  photoStrip: { gap: 12, paddingVertical: 4 },
  historyItem: { width: 100, gap: 7, backgroundColor: colors.white, padding: 6 },
  historyCaption: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  historyScore: { fontFamily: mono, color: colors.ink, fontSize: 12, fontWeight: '700' },
  resultCard: { paddingVertical: 18, borderTopWidth: 3, borderBottomWidth: 1, borderColor: colors.ink, gap: 18 },
  answer: { fontFamily: serif, color: colors.ink, fontSize: 44, fontWeight: '700', letterSpacing: -1 },
  resultBadge: { alignSelf: 'flex-start', borderBottomWidth: 1, borderColor: colors.orange, paddingVertical: 5, marginTop: -9 },
  resultBadgeText: { color: colors.orange, fontSize: 11, letterSpacing: 1 },
  resultStats: { flexDirection: 'row', borderTopWidth: 1, borderBottomWidth: 1, borderColor: colors.line, paddingVertical: 16, gap: 8 },
  metric: { flex: 1, gap: 6 },
  metricValue: { fontFamily: mono, fontSize: 20, fontWeight: '700', color: colors.ink },
  journey: { paddingVertical: 4, gap: 10 },
  journeyText: { fontFamily: mono, fontSize: 13, lineHeight: 24, color: colors.muted },
  shareNote: { fontSize: 10, textAlign: 'center', color: colors.muted, marginTop: -7, lineHeight: 17 },
  nextPuzzle: { gap: 7, borderTopWidth: 1, borderStyle: 'dashed', borderColor: colors.line, paddingTop: 16 },
  nextTime: { fontFamily: mono, fontSize: 23, color: colors.ink, fontWeight: '700' },
  footer: { paddingTop: 8, gap: 7 },
  footerText: { fontFamily: serif, fontSize: 12, color: colors.ink },
  deskFooter: { fontFamily: serif, color: '#8E9485', textAlign: 'center', fontSize: 11, letterSpacing: 2, marginTop: 20 },
  loading: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 24, padding: 32 },
});
