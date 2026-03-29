import { useState, useRef, useEffect, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  Platform,
  Modal,
  ActivityIndicator,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import type { CameraType } from "expo-camera";
import * as ImagePicker from "expo-image-picker";
import * as Speech from "expo-speech";
import { useVideoPlayer, VideoView } from "expo-video";
import * as WebBrowser from "expo-web-browser";
import { Ionicons } from "@expo/vector-icons";
import {
  predictSign,
  predictSentence,
  type PredictionResult,
  type SentencePredictionResult,
} from "../services/api";
import AsyncStorage from "@react-native-async-storage/async-storage";

const RECORD_DURATION_S = 5;
const HISTORY_KEY = "eyehearu_history";

const BRAND = {
  teal: "#0D9488",
  tealDark: "#0F766E",
  coral: "#F97066",
  bg: "#F0FDFA",
  card: "#FFFFFF",
  textPrimary: "#134E4A",
  textSecondary: "#5F7572",
  textMuted: "#94A3B8",
};

type TranslateMode = "single" | "sentence";

function signToSlug(sign: string): string {
  return sign
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [translateMode, setTranslateMode] = useState<TranslateMode>("single");
  const [sentenceUris, setSentenceUris] = useState<string[]>([]);
  const [sentenceResult, setSentenceResult] =
    useState<SentencePredictionResult | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [topK, setTopK] = useState<{ sign: string; confidence: number }[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [facing, setFacing] = useState<CameraType>("front");
  const [countdown, setCountdown] = useState<number>(0);
  const [videoModal, setVideoModal] = useState<{ sign: string; urls: string[]; index: number } | null>(null);
  const [videoLoading, setVideoLoading] = useState(true);
  const [videoError, setVideoError] = useState(false);

  const currentVideoUrl = videoModal ? videoModal.urls[videoModal.index] : null;
  const player = useVideoPlayer(currentVideoUrl, (p) => {
    p.loop = true;
    p.play();
  });

  const tryNextVideoSource = useCallback(() => {
    setVideoModal((prev) => {
      /* istanbul ignore next — defensive if the player fires after the modal was cleared */
      if (!prev || prev.urls.length === 0) return prev;
      const next = prev.index + 1;
      if (next < prev.urls.length) return { ...prev, index: next };
      queueMicrotask(() => {
        setVideoError(true);
        setVideoLoading(false);
      });
      return prev;
    });
  }, []);

  useEffect(() => {
    if (!currentVideoUrl) return;
    setVideoLoading(true);
    setVideoError(false);
  }, [currentVideoUrl]);

  useEffect(() => {
    if (!currentVideoUrl || !player) return;
    const statusSub = player.addListener("statusChange", (payload) => {
      if (payload.status === "readyToPlay") setVideoLoading(false);
      if (payload.status === "error") tryNextVideoSource();
    });
    return () => statusSub.remove();
  }, [currentVideoUrl, player, tryNextVideoSource]);

  const cameraRef = useRef<CameraView>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const busy = isRecording || isProcessing;

  /* Pulse animation while recording */
  useEffect(() => {
    if (!isRecording) return;
    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.5,
          duration: 500,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 500,
          useNativeDriver: true,
        }),
      ])
    );
    anim.start();
    return () => anim.stop();
  }, [isRecording, pulseAnim]);

  /* Countdown timer while recording */
  useEffect(() => {
    if (!isRecording) {
      if (countdownRef.current) clearInterval(countdownRef.current);
      setCountdown(0);
      return;
    }
    setCountdown(RECORD_DURATION_S);
    countdownRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(countdownRef.current!);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => {
      clearInterval(countdownRef.current!);
    };
  }, [isRecording]);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionIcon}>{"\u{1F4F7}"}</Text>
        <Text style={styles.permissionTitle}>Camera Access Needed</Text>
        <Text style={styles.permissionText}>
          Eye Hear U uses the device camera to record ASL signs for translation.
        </Text>
        <TouchableOpacity style={styles.grantButton} onPress={requestPermission}>
          <Text style={styles.grantButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const toggleCamera = () => {
    setFacing((prev) => (prev === "front" ? "back" : "front"));
  };

  const handlePredictionResult = async (result: PredictionResult) => {
    setSentenceResult(null);
    setPrediction(result.sign);
    setConfidence(result.confidence);
    setTopK(result.top_k || []);

    if (result.sign) {
      await saveToHistory(result);
      if (result.confidence > 0.3) {
        Speech.speak(result.sign, { language: "en-US", rate: 0.9 });
      }
    }
  };

  const handleSentenceResult = async (result: SentencePredictionResult) => {
    setPrediction(null);
    setTopK([]);
    setConfidence(0);
    setSentenceResult(result);
    if (result.english) {
      await saveSentenceToHistory(result);
      Speech.speak(result.english, { language: "en-US", rate: 0.9 });
    }
  };

  const recordAndPredict = async () => {
    /* istanbul ignore next — CameraView always mounts ref; UI prevents calls while busy */
    if (!cameraRef.current || busy) return;

    setIsRecording(true);
    setPrediction(null);
    setTopK([]);
    setSentenceResult(null);
    setErrorMessage(null);

    try {
      const video = await cameraRef.current.recordAsync({
        maxDuration: RECORD_DURATION_S,
      });

      setIsRecording(false);
      if (!video?.uri) return;

      if (translateMode === "sentence") {
        setSentenceUris((prev) => [...prev, video.uri]);
        return;
      }

      setIsProcessing(true);
      const result: PredictionResult = await predictSign(video.uri);
      await handlePredictionResult(result);
    } catch (error) {
      console.error("Prediction failed:", error);
      const msg = error instanceof Error ? error.message : String(error);
      setErrorMessage(msg);
      setPrediction(null);
      setConfidence(0);
      setTopK([]);
      setIsRecording(false);
    } finally {
      setIsProcessing(false);
    }
  };

  const translateSentence = async () => {
    /* istanbul ignore next — Translate is disabled unless there are clips and the user is not busy */
    if (busy || sentenceUris.length === 0) return;
    setErrorMessage(null);
    setSentenceResult(null);
    setPrediction(null);
    setTopK([]);
    setIsProcessing(true);
    try {
      const result = await predictSentence(sentenceUris);
      await handleSentenceResult(result);
    } catch (error) {
      console.error("Sentence prediction failed:", error);
      const msg = error instanceof Error ? error.message : String(error);
      setErrorMessage(msg);
      setSentenceResult(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const pickAndPredict = async () => {
    /* istanbul ignore next — upload control is disabled while busy */
    if (busy) return;

    setPrediction(null);
    setTopK([]);
    setSentenceResult(null);
    setErrorMessage(null);

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ["videos"],
        quality: 1,
        videoMaxDuration: 10,
      });

      if (result.canceled || !result.assets?.[0]?.uri) return;

      if (translateMode === "sentence") {
        setSentenceUris((prev) => [...prev, result.assets[0].uri]);
        return;
      }

      setIsProcessing(true);
      const predResult: PredictionResult = await predictSign(result.assets[0].uri);
      await handlePredictionResult(predResult);
    } catch (error) {
      console.error("Upload prediction failed:", error);
      const msg = error instanceof Error ? error.message : String(error);
      setErrorMessage(msg);
      setPrediction(null);
      setConfidence(0);
      setTopK([]);
    } finally {
      setIsProcessing(false);
    }
  };

  const stopRecording = () => {
    /* istanbul ignore next */
    if (cameraRef.current && isRecording) {
      cameraRef.current.stopRecording();
    }
  };

  const saveToHistory = async (result: PredictionResult) => {
    try {
      const raw = await AsyncStorage.getItem(HISTORY_KEY);
      const history = raw ? JSON.parse(raw) : [];
      history.unshift({
        id: Date.now().toString(),
        sign: result.sign,
        confidence: result.confidence,
        timestamp: new Date().toISOString(),
      });
      if (history.length > 100) history.length = 100;
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch {}
  };

  const saveSentenceToHistory = async (result: SentencePredictionResult) => {
    try {
      const raw = await AsyncStorage.getItem(HISTORY_KEY);
      const history = raw ? JSON.parse(raw) : [];
      const beamScore = result.beam[0]?.score ?? 0;
      const conf = Math.min(1, Math.max(0, 1 / (1 + Math.exp(-beamScore))));
      history.unshift({
        id: Date.now().toString(),
        sign: result.english,
        confidence: conf,
        timestamp: new Date().toISOString(),
        sentence: true,
        glosses: result.best_glosses,
      });
      if (history.length > 100) history.length = 100;
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch {}
  };

  const speakPrediction = () => {
    const text = sentenceResult?.english ?? prediction;
    /* istanbul ignore next — buttons that call this are only shown when there is text to speak */
    if (text) {
      Speech.speak(text, { language: "en-US", rate: 0.9 });
    }
  };

  const setMode = (next: TranslateMode) => {
    setTranslateMode(next);
    setSentenceUris([]);
    setSentenceResult(null);
    setPrediction(null);
    setTopK([]);
    setErrorMessage(null);
  };

  const closeVideoModal = () => {
    setVideoModal(null);
    setVideoLoading(true);
    setVideoError(false);
  };

  const openVideoInBrowser = async () => {
    /* istanbul ignore next — modal buttons only render while videoModal is set */
    if (!videoModal) return;
    const slug = signToSlug(videoModal.sign);
    if (!slug) return;
    await WebBrowser.openBrowserAsync(
      `https://www.signasl.org/sign/${encodeURIComponent(slug)}`
    );
  };

  const openSignVideo = async (sign: string) => {
    /* istanbul ignore next — UI always passes a non-empty gloss label */
    if (!sign) return;
    setVideoLoading(true);
    setVideoError(false);
    const slug = signToSlug(sign);
    if (!slug) {
      setVideoModal({ sign, urls: [], index: 0 });
      setVideoError(true);
      setVideoLoading(false);
      return;
    }
    try {
      const res = await fetch(
        `https://www.signasl.org/sign/${encodeURIComponent(slug)}`
      );
      if (!res.ok) throw new Error("Sign page not found");
      const html = await res.text();
      const seen = new Set<string>();
      const urls: string[] = [];
      const re = /https?:\/\/[^"'\\\s<>]+\.mp4/gi;
      let m: RegExpExecArray | null = re.exec(html);
      while (m !== null) {
        const u = m[0];
        if (!seen.has(u)) {
          seen.add(u);
          urls.push(u);
        }
        m = re.exec(html);
      }
      if (urls.length === 0) {
        setVideoModal({ sign, urls: [], index: 0 });
        setVideoError(true);
        setVideoLoading(false);
        return;
      }
      setVideoModal({ sign, urls, index: 0 });
    } catch {
      setVideoModal({ sign, urls: [], index: 0 });
      setVideoError(true);
      setVideoLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing={facing} mode="video" />

      {/* Recording guidance overlay — visible when idle */}
      {!isRecording &&
        !isProcessing &&
        !prediction &&
        !errorMessage &&
        !sentenceResult &&
        !(translateMode === "sentence" && sentenceUris.length > 0) && (
        <View style={styles.guidanceOverlay} pointerEvents="none">
          <View style={styles.guidanceSilhouette}>
            <View style={styles.silhouetteHead} />
            <View style={styles.silhouetteBody}>
              {/* Left hand */}
              <View style={[styles.silhouetteHand, styles.handLeft]} />
              {/* Right hand */}
              <View style={[styles.silhouetteHand, styles.handRight]} />
            </View>
          </View>
          <Text style={styles.guidanceText}>
            {translateMode === "sentence" ? (
              "Record each sign one at a time, then Translate"
            ) : (
              <>
                Center yourself in the frame{"\n"}Keep hands visible
              </>
            )}
          </Text>
        </View>
      )}

      {/* Camera toggle + upload + mode */}
      <View
        testID="camera-top-controls"
        style={[
          styles.topControls,
          { top: Platform.OS === "ios" ? 16 : 12 },
        ]}
      >
        <TouchableOpacity
          style={styles.topButton}
          onPress={toggleCamera}
          disabled={busy}
        >
          <Ionicons name="camera-reverse-outline" size={22} color="#fff" />
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.topButton}
          onPress={pickAndPredict}
          disabled={busy}
        >
          <Ionicons name="cloud-upload-outline" size={22} color="#fff" />
        </TouchableOpacity>
      </View>
      <View
        testID="camera-mode-row"
        style={[
          styles.modeRow,
          { top: Platform.OS === "ios" ? 64 : 58 },
        ]}
      >
        <TouchableOpacity
          style={[
            styles.modeChip,
            translateMode === "single" && styles.modeChipActive,
          ]}
          onPress={() => setMode("single")}
          disabled={busy}
        >
          <Text
            style={[
              styles.modeChipText,
              translateMode === "single" && styles.modeChipTextActive,
            ]}
          >
            Single sign
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.modeChip,
            translateMode === "sentence" && styles.modeChipActive,
          ]}
          onPress={() => setMode("sentence")}
          disabled={busy}
        >
          <Text
            style={[
              styles.modeChipText,
              translateMode === "sentence" && styles.modeChipTextActive,
            ]}
          >
            Multi-sign
          </Text>
        </TouchableOpacity>
      </View>

      {/* Recording overlay with countdown */}
      {isRecording && (
        <View
          testID="camera-recording-overlay"
          style={[
            styles.recordingOverlay,
            { top: Platform.OS === "ios" ? 16 : 12 },
          ]}
        >
          <Animated.View
            style={[styles.recordDot, { transform: [{ scale: pulseAnim }] }]}
          />
          <Text style={styles.recordingText}>
            Recording{countdown > 0 ? ` ${countdown}s` : ""}
          </Text>
        </View>
      )}

      <View style={styles.resultContainer}>
        {isProcessing ? (
          <View style={styles.processingRow}>
            <Text style={styles.processingEmoji}>{"\u{1F914}"}</Text>
            <Text style={styles.processingText}>Analyzing sign...</Text>
          </View>
        ) : errorMessage ? (
          <View style={styles.errorBox}>
            <Text style={styles.errorTitle}>Request failed</Text>
            <Text style={styles.errorDetail}>{errorMessage}</Text>
          </View>
        ) : sentenceResult ? (
          <>
            <Text style={styles.predictionLabel}>Sentence</Text>
            <Text style={styles.sentenceText}>{sentenceResult.english}</Text>
            <Text style={styles.confidenceText}>
              {sentenceResult.best_glosses.join(" · ")}
            </Text>
            <TouchableOpacity style={styles.speakButton} onPress={speakPrediction}>
              <Text style={styles.speakButtonText}>
                {"\u{1F50A}"} Speak Again
              </Text>
            </TouchableOpacity>
          </>
        ) : translateMode === "sentence" && sentenceUris.length > 0 ? (
          <View style={styles.segmentHint}>
            <Text style={styles.segmentHintTitle}>
              {sentenceUris.length} clip(s) — tap Translate below
            </Text>
            <TouchableOpacity
              onPress={() => {
                setSentenceUris([]);
                setSentenceResult(null);
              }}
              style={styles.clearSegmentsBtn}
            >
              <Text style={styles.clearSegmentsText}>Clear clips</Text>
            </TouchableOpacity>
          </View>
        ) : prediction ? (
          <>
            <Text style={styles.predictionLabel}>Detected Sign</Text>
            <TouchableOpacity onPress={() => openSignVideo(prediction)} activeOpacity={0.6}>
              <View style={styles.predictionTouchable}>
                <Text style={styles.predictionText}>{prediction}</Text>
                <Ionicons name="videocam-outline" size={20} color={BRAND.teal} style={styles.videoIcon} />
              </View>
            </TouchableOpacity>
            <View style={styles.confidenceBar}>
              <View
                style={[
                  styles.confidenceFill,
                  { width: `${Math.min(confidence * 100, 100)}%` },
                ]}
              />
            </View>
            <Text style={styles.confidenceText}>
              {(confidence * 100).toFixed(1)}% confidence
            </Text>
            {topK.length > 1 && (
              <View style={styles.topKRow}>
                {topK.slice(1, 4).map((item, i) => (
                  <TouchableOpacity key={i} style={styles.topKChip} onPress={() => openSignVideo(item.sign)} activeOpacity={0.6}>
                    <Ionicons name="play-circle-outline" size={14} color={BRAND.tealDark} />
                    <Text style={styles.topKText}>
                      {item.sign} {(item.confidence * 100).toFixed(0)}%
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            )}
            <TouchableOpacity style={styles.speakButton} onPress={speakPrediction}>
              <Text style={styles.speakButtonText}>
                {"\u{1F50A}"} Speak Again
              </Text>
            </TouchableOpacity>
          </>
        ) : (
          <Text style={styles.instructionText}>
            {translateMode === "sentence"
              ? "\u{1F91F} Multi-sign: record each gloss, then Translate"
              : "\u{1F91F} Sign in front of the camera, then tap Record"}
          </Text>
        )}
      </View>

      <View style={styles.bottomControls}>
        {translateMode === "single" ? (
          <TouchableOpacity
            style={[
              styles.captureButton,
              isRecording && styles.captureButtonRecording,
              isProcessing && styles.captureButtonDisabled,
            ]}
            onPress={isRecording ? stopRecording : recordAndPredict}
            disabled={isProcessing}
            activeOpacity={0.8}
          >
            {isRecording && <View style={styles.stopIcon} />}
            <Text style={styles.captureButtonText}>
              {isProcessing
                ? "Processing..."
                : isRecording
                  ? "  Stop"
                  : "Record Sign"}
            </Text>
          </TouchableOpacity>
        ) : (
          <View style={styles.sentenceBottomRow}>
            <TouchableOpacity
              style={[
                styles.captureButtonHalf,
                isRecording && styles.captureButtonRecording,
                isProcessing && styles.captureButtonDisabled,
              ]}
              onPress={isRecording ? stopRecording : recordAndPredict}
              disabled={isProcessing}
              activeOpacity={0.8}
            >
              {isRecording && <View style={styles.stopIcon} />}
              <Text style={styles.captureButtonText}>
                {isProcessing
                  ? "…"
                  : isRecording
                    ? " Stop"
                    : "Add sign"}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.captureButtonHalf,
                styles.translateButton,
                (sentenceUris.length === 0 || isProcessing) &&
                  styles.captureButtonDisabled,
              ]}
              onPress={translateSentence}
              disabled={sentenceUris.length === 0 || isProcessing || isRecording}
              activeOpacity={0.8}
            >
              <Text style={styles.captureButtonText}>
                {isProcessing ? "…" : "Translate"}
              </Text>
            </TouchableOpacity>
          </View>
        )}
      </View>

      <Modal
        testID="sign-video-modal"
        visible={!!videoModal}
        transparent
        animationType="fade"
        onRequestClose={closeVideoModal}
      >
        <View style={styles.modalBackdrop}>
          <View style={styles.modalCard}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>
                {videoModal?.sign}
              </Text>
              <TouchableOpacity onPress={closeVideoModal} hitSlop={12}>
                <Ionicons name="close-circle" size={28} color={BRAND.textMuted} />
              </TouchableOpacity>
            </View>

            <View style={styles.videoWrapper}>
              {videoLoading && !videoError && (
                <View style={styles.videoLoading}>
                  <ActivityIndicator size="large" color={BRAND.teal} />
                  <Text style={styles.videoLoadingText}>Loading video...</Text>
                </View>
              )}
              {videoError ? (
                <View style={styles.videoErrorContainer}>
                  <Ionicons name="alert-circle-outline" size={40} color={BRAND.textMuted} />
                  <Text style={styles.videoErrorText}>
                    Video unavailable
                  </Text>
                  <TouchableOpacity style={styles.browserFallback} onPress={openVideoInBrowser}>
                    <Ionicons name="open-outline" size={16} color="#fff" />
                    <Text style={styles.browserFallbackText}>Open in Browser</Text>
                  </TouchableOpacity>
                </View>
              ) : videoModal ? (
                <VideoView
                  player={player}
                  style={styles.video}
                  contentFit="contain"
                  nativeControls
                />
              ) : null}
            </View>

            <View style={styles.modalFooter}>
              <TouchableOpacity style={styles.browserLink} onPress={openVideoInBrowser}>
                <Ionicons name="open-outline" size={14} color={BRAND.teal} />
                <Text style={styles.browserLinkText}>More videos on SignASL.org</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  camera: { flex: 1 },
  /* --- Guidance overlay (larger silhouette with hands) --- */
  guidanceOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: "center",
    alignItems: "center",
  },
  guidanceSilhouette: {
    alignItems: "center",
    opacity: 0.35,
    marginBottom: 16,
  },
  silhouetteHead: {
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 2.5,
    borderColor: "#fff",
    marginBottom: 6,
  },
  silhouetteBody: {
    width: 140,
    height: 110,
    borderTopLeftRadius: 56,
    borderTopRightRadius: 56,
    borderWidth: 2.5,
    borderColor: "#fff",
    borderBottomWidth: 0,
    position: "relative",
  },
  silhouetteHand: {
    width: 32,
    height: 32,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: "#fff",
    position: "absolute",
    bottom: 10,
  },
  handLeft: {
    left: -30,
  },
  handRight: {
    right: -30,
  },
  guidanceText: {
    color: "rgba(255,255,255,0.8)",
    fontSize: 15,
    textAlign: "center",
    lineHeight: 22,
    backgroundColor: "rgba(0,0,0,0.4)",
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 14,
  },
  /* --- Top controls (camera toggle + upload) --- */
  topControls: {
    position: "absolute",
    right: 16,
    flexDirection: "row",
    gap: 10,
  },
  topButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "center",
    alignItems: "center",
  },
  modeRow: {
    position: "absolute",
    left: 16,
    right: 16,
    flexDirection: "row",
    justifyContent: "center",
    gap: 8,
  },
  modeChip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: "rgba(0,0,0,0.45)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.3)",
  },
  modeChipActive: {
    backgroundColor: BRAND.teal,
    borderColor: BRAND.teal,
  },
  modeChipText: {
    color: "rgba(255,255,255,0.9)",
    fontSize: 13,
    fontWeight: "600",
  },
  modeChipTextActive: { color: "#fff" },
  segmentHint: {
    alignItems: "center",
    alignSelf: "stretch",
  },
  segmentHintTitle: {
    fontSize: 15,
    fontWeight: "600",
    color: BRAND.textSecondary,
    marginBottom: 8,
    textAlign: "center",
  },
  clearSegmentsBtn: {
    paddingVertical: 8,
    paddingHorizontal: 18,
  },
  clearSegmentsText: {
    color: BRAND.teal,
    fontWeight: "600",
    fontSize: 14,
  },
  /* --- Recording overlay --- */
  recordingOverlay: {
    position: "absolute",
    left: 16,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 24,
  },
  recordDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: "#FF3B30",
    marginRight: 8,
  },
  recordingText: { color: "#fff", fontSize: 14, fontWeight: "600" },
  /* --- Result container --- */
  resultContainer: {
    backgroundColor: "#fff",
    paddingVertical: 18,
    paddingHorizontal: 24,
    alignItems: "center",
    minHeight: 140,
    justifyContent: "center",
  },
  processingRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  processingEmoji: { fontSize: 24 },
  processingText: { fontSize: 16, color: BRAND.textMuted },
  predictionLabel: { fontSize: 12, color: BRAND.textMuted, marginBottom: 2, textTransform: "uppercase", letterSpacing: 1 },
  predictionTouchable: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    marginBottom: 6,
  },
  predictionText: {
    fontSize: 38,
    fontWeight: "800",
    color: BRAND.teal,
    textTransform: "capitalize",
    textDecorationLine: "underline",
    textDecorationStyle: "dotted",
    textDecorationColor: BRAND.teal,
  },
  videoIcon: {
    marginTop: 4,
  },
  sentenceText: {
    fontSize: 38,
    fontWeight: "800",
    color: BRAND.teal,
    marginBottom: 6,
  },
  confidenceBar: {
    width: "60%",
    height: 6,
    backgroundColor: "#E2E8F0",
    borderRadius: 3,
    marginBottom: 4,
    overflow: "hidden",
  },
  confidenceFill: {
    height: "100%",
    backgroundColor: BRAND.teal,
    borderRadius: 3,
  },
  confidenceText: { fontSize: 13, color: BRAND.textSecondary, marginBottom: 8 },
  topKRow: { flexDirection: "row", gap: 8, marginBottom: 10, flexWrap: "wrap", justifyContent: "center" },
  topKChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    backgroundColor: BRAND.bg,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#D1E7E5",
  },
  topKText: { fontSize: 12, color: BRAND.tealDark },
  errorBox: { alignSelf: "stretch", maxHeight: 220 },
  errorTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: BRAND.coral,
    textAlign: "center",
    marginBottom: 8,
  },
  errorDetail: {
    fontSize: 12,
    color: BRAND.textSecondary,
    textAlign: "left",
    lineHeight: 18,
  },
  instructionText: { fontSize: 16, color: BRAND.textMuted, textAlign: "center" },
  /* --- Bottom controls --- */
  bottomControls: {
    flexDirection: "row",
  },
  sentenceBottomRow: {
    flexDirection: "row",
    width: "100%",
  },
  captureButtonHalf: {
    flex: 1,
    backgroundColor: BRAND.teal,
    paddingVertical: 18,
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "center",
  },
  translateButton: {
    backgroundColor: BRAND.tealDark,
  },
  captureButton: {
    flex: 1,
    backgroundColor: BRAND.teal,
    paddingVertical: 18,
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "center",
  },
  captureButtonRecording: { backgroundColor: "#DC2626" },
  captureButtonDisabled: { backgroundColor: "#94D3CF" },
  stopIcon: {
    width: 16,
    height: 16,
    borderRadius: 3,
    backgroundColor: "#fff",
  },
  captureButtonText: { color: "#fff", fontSize: 20, fontWeight: "bold" },
  speakButton: {
    backgroundColor: BRAND.bg,
    paddingVertical: 8,
    paddingHorizontal: 18,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#D1E7E5",
  },
  speakButtonText: { color: BRAND.teal, fontWeight: "600", fontSize: 14 },
  permissionContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 32,
    backgroundColor: BRAND.bg,
  },
  permissionIcon: { fontSize: 48, marginBottom: 16 },
  permissionTitle: {
    fontSize: 22,
    fontWeight: "bold",
    color: BRAND.textPrimary,
    marginBottom: 10,
  },
  permissionText: {
    fontSize: 16,
    color: BRAND.textSecondary,
    textAlign: "center",
    marginBottom: 24,
    lineHeight: 22,
  },
  grantButton: {
    backgroundColor: BRAND.teal,
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 14,
  },
  grantButtonText: { color: "#fff", fontSize: 17, fontWeight: "600" },
  /* --- Video modal --- */
  modalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.7)",
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  modalCard: {
    width: "100%",
    maxWidth: 400,
    backgroundColor: "#fff",
    borderRadius: 20,
    overflow: "hidden",
  },
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 10,
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: "700",
    color: BRAND.textPrimary,
    textTransform: "capitalize",
  },
  videoWrapper: {
    width: "100%",
    aspectRatio: 4 / 3,
    backgroundColor: "#000",
    position: "relative",
  },
  video: {
    width: "100%",
    height: "100%",
  },
  videoLoading: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000",
    zIndex: 1,
  },
  videoLoadingText: {
    color: "#aaa",
    fontSize: 13,
    marginTop: 8,
  },
  videoErrorContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    gap: 10,
  },
  videoErrorText: {
    color: "#aaa",
    fontSize: 15,
  },
  browserFallback: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: BRAND.teal,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 10,
    marginTop: 6,
  },
  browserFallbackText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 14,
  },
  modalFooter: {
    alignItems: "center",
    paddingVertical: 14,
  },
  browserLink: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
  },
  browserLinkText: {
    color: BRAND.teal,
    fontSize: 13,
    fontWeight: "500",
  },
});
