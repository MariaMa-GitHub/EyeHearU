import { useState, useRef, useEffect } from "react";
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
import { predictSign, type PredictionResult } from "../services/api";
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

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
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

  useEffect(() => {
    if (!currentVideoUrl || !player) return;
    const statusSub = player.addListener("statusChange", (payload) => {
      if (payload.status === "readyToPlay") setVideoLoading(false);
      if (payload.status === "error") tryNextVideoSource();
    });
    return () => statusSub.remove();
  }, [currentVideoUrl, player]);

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
          if (countdownRef.current) clearInterval(countdownRef.current);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => {
      if (countdownRef.current) clearInterval(countdownRef.current);
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

  const recordAndPredict = async () => {
    if (!cameraRef.current || busy) return;

    setIsRecording(true);
    setPrediction(null);
    setTopK([]);
    setErrorMessage(null);

    try {
      const video = await cameraRef.current.recordAsync({
        maxDuration: RECORD_DURATION_S,
      });

      setIsRecording(false);
      if (!video?.uri) return;

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

  const pickAndPredict = async () => {
    if (busy) return;

    setPrediction(null);
    setTopK([]);
    setErrorMessage(null);

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ["videos"],
        quality: 1,
        videoMaxDuration: 10,
      });

      if (result.canceled || !result.assets?.[0]?.uri) return;

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

  const signToSlug = (sign: string) =>
    sign.trim().toLowerCase().replace(/\s+/g, "-");

  const buildVideoUrls = (sign: string): string[] => {
    const slug = signToSlug(sign);
    const encoded = encodeURIComponent(sign.trim().replace(/\b\w/g, c => c.toUpperCase()));
    const base = "https://media.signbsl.com/videos/asl";
    return [
      `${base}/startasl/mp4/${slug}.mp4`,
      `${base}/elementalaslconcepts/mp4/${slug}.mp4`,
      `${base}/mariekatzenbachschool/mp4/${slug}.mp4`,
      `${base}/mariekatzenbachschool/mp4/${encoded}.mp4`,
    ];
  };

  const openSignVideo = (sign: string) => {
    const urls = buildVideoUrls(sign);
    setVideoLoading(true);
    setVideoError(false);
    setVideoModal({ sign, urls, index: 0 });
  };

  const tryNextVideoSource = () => {
    if (!videoModal) return;
    const next = videoModal.index + 1;
    if (next < videoModal.urls.length) {
      setVideoLoading(true);
      setVideoModal({ ...videoModal, index: next });
    } else {
      setVideoLoading(false);
      setVideoError(true);
    }
  };

  const closeVideoModal = () => {
    setVideoModal(null);
    setVideoLoading(true);
    setVideoError(false);
  };

  const openVideoInBrowser = async () => {
    if (!videoModal) return;
    closeVideoModal();
    const slug = signToSlug(videoModal.sign);
    await WebBrowser.openBrowserAsync(`https://signasl.org/sign/${slug}`);
  };

  const speakPrediction = () => {
    if (prediction) {
      Speech.speak(prediction, { language: "en-US", rate: 0.9 });
    }
  };

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing={facing} mode="video" />

      {/* Recording guidance overlay — visible when idle */}
      {!isRecording && !isProcessing && !prediction && !errorMessage && (
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
            Center yourself in the frame{"\n"}Keep hands visible
          </Text>
        </View>
      )}

      {/* Camera toggle + upload buttons */}
      <View style={styles.topControls}>
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

      {/* Recording overlay with countdown */}
      {isRecording && (
        <View style={styles.recordingOverlay}>
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
            {"\u{1F91F}"} Sign in front of the camera, then tap Record
          </Text>
        )}
      </View>

      <View style={styles.bottomControls}>
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
      </View>

      <Modal
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
    top: Platform.OS === "ios" ? 16 : 12,
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
  /* --- Recording overlay --- */
  recordingOverlay: {
    position: "absolute",
    top: Platform.OS === "ios" ? 16 : 12,
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
