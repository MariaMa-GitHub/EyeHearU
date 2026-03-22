import { useState, useRef, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  Platform,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as Speech from "expo-speech";
import { predictSign, type PredictionResult } from "../services/api";
import AsyncStorage from "@react-native-async-storage/async-storage";

const RECORD_DURATION_MS = 3000;
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
  const cameraRef = useRef<CameraView>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;

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

  const recordAndPredict = async () => {
    if (!cameraRef.current || isRecording || isProcessing) return;

    setIsRecording(true);
    setPrediction(null);
    setTopK([]);
    setErrorMessage(null);

    try {
      const video = await cameraRef.current.recordAsync({
        maxDuration: RECORD_DURATION_MS / 1000,
      });

      setIsRecording(false);
      if (!video?.uri) return;

      setIsProcessing(true);
      const result: PredictionResult = await predictSign(video.uri);
      setPrediction(result.sign);
      setConfidence(result.confidence);
      setTopK(result.top_k || []);

      if (result.sign) {
        await saveToHistory(result);
        if (result.confidence > 0.3) {
          Speech.speak(result.sign, { language: "en-US", rate: 0.9 });
        }
      }
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

  const speakPrediction = () => {
    if (prediction) {
      Speech.speak(prediction, { language: "en-US", rate: 0.9 });
    }
  };

  const busy = isRecording || isProcessing;

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing="front" mode="video" />
      {isRecording && (
        <View style={styles.recordingOverlay}>
          <Animated.View
            style={[styles.recordDot, { transform: [{ scale: pulseAnim }] }]}
          />
          <Text style={styles.recordingText}>Recording… hold the sign steady</Text>
        </View>
      )}

      <View style={styles.resultContainer}>
        {isProcessing ? (
          <View style={styles.processingRow}>
            <Text style={styles.processingEmoji}>{"\u{1F914}"}</Text>
            <Text style={styles.processingText}>Analyzing sign…</Text>
          </View>
        ) : errorMessage ? (
          <View style={styles.errorBox}>
            <Text style={styles.errorTitle}>Request failed</Text>
            <Text style={styles.errorDetail}>{errorMessage}</Text>
          </View>
        ) : prediction ? (
          <>
            <Text style={styles.predictionLabel}>Detected Sign</Text>
            <Text style={styles.predictionText}>{prediction}</Text>
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
                  <View key={i} style={styles.topKChip}>
                    <Text style={styles.topKText}>
                      {item.sign} {(item.confidence * 100).toFixed(0)}%
                    </Text>
                  </View>
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
        {isRecording && (
          <View style={styles.stopIcon} />
        )}
        <Text style={styles.captureButtonText}>
          {isProcessing
            ? "Processing…"
            : isRecording
            ? "  Stop"
            : "Record Sign"}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  camera: { flex: 1 },
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
  predictionText: {
    fontSize: 38,
    fontWeight: "800",
    color: BRAND.teal,
    marginBottom: 6,
    textTransform: "capitalize",
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
    backgroundColor: BRAND.bg,
    paddingHorizontal: 10,
    paddingVertical: 4,
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
  captureButton: {
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
});
