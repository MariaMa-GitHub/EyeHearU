import { useState, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Alert,
} from "react-native";
import { useFocusEffect } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";

const HISTORY_KEY = "eyehearu_history";

const BRAND = {
  teal: "#0D9488",
  coral: "#F97066",
  bg: "#F0FDFA",
  card: "#FFFFFF",
  textPrimary: "#134E4A",
  textSecondary: "#5F7572",
  textMuted: "#94A3B8",
};

interface HistoryItem {
  id: string;
  sign: string;
  confidence: number;
  timestamp: string;
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export default function HistoryScreen() {
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useFocusEffect(
    useCallback(() => {
      loadHistory();
    }, [])
  );

  const loadHistory = async () => {
    try {
      const raw = await AsyncStorage.getItem(HISTORY_KEY);
      setHistory(raw ? JSON.parse(raw) : []);
    } catch {
      setHistory([]);
    }
  };

  const clearHistory = () => {
    Alert.alert("Clear History", "Remove all saved translations?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Clear",
        style: "destructive",
        onPress: async () => {
          await AsyncStorage.removeItem(HISTORY_KEY);
          setHistory([]);
        },
      },
    ]);
  };

  return (
    <View style={styles.container}>
      {history.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyIcon}>{"\u{1F91F}"}</Text>
          <Text style={styles.emptyText}>No translations yet</Text>
          <Text style={styles.emptySubtext}>
            Go to the camera to start translating ASL signs.
          </Text>
        </View>
      ) : (
        <>
          <FlatList
            data={history}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.list}
            renderItem={({ item }) => (
              <View style={styles.card}>
                <View style={styles.cardHeader}>
                  <Text style={styles.sign}>{item.sign}</Text>
                  <Text style={styles.timestamp}>{timeAgo(item.timestamp)}</Text>
                </View>
                <View style={styles.cardFooter}>
                  <View style={styles.confidenceBar}>
                    <View
                      style={[
                        styles.confidenceFill,
                        { width: `${Math.min(item.confidence * 100, 100)}%` },
                      ]}
                    />
                  </View>
                  <Text style={styles.confidence}>
                    {(item.confidence * 100).toFixed(0)}%
                  </Text>
                </View>
              </View>
            )}
          />
          <TouchableOpacity style={styles.clearButton} onPress={clearHistory}>
            <Text style={styles.clearButtonText}>Clear History</Text>
          </TouchableOpacity>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: BRAND.bg },
  list: { padding: 16, gap: 10, paddingBottom: 80 },
  card: {
    backgroundColor: BRAND.card,
    padding: 16,
    borderRadius: 14,
    shadowColor: "#0D9488",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 2,
  },
  cardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "baseline",
    marginBottom: 8,
  },
  sign: {
    fontSize: 22,
    fontWeight: "700",
    color: BRAND.teal,
    textTransform: "capitalize",
  },
  timestamp: { color: BRAND.textMuted, fontSize: 13 },
  cardFooter: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  confidenceBar: {
    flex: 1,
    height: 5,
    backgroundColor: "#E2E8F0",
    borderRadius: 3,
    overflow: "hidden",
  },
  confidenceFill: {
    height: "100%",
    backgroundColor: BRAND.teal,
    borderRadius: 3,
  },
  confidence: { color: BRAND.textSecondary, fontSize: 13, fontWeight: "600", minWidth: 36, textAlign: "right" },
  empty: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 32,
  },
  emptyIcon: { fontSize: 56, marginBottom: 16 },
  emptyText: { fontSize: 20, fontWeight: "700", color: BRAND.textPrimary, marginBottom: 8 },
  emptySubtext: { fontSize: 15, color: BRAND.textMuted, textAlign: "center", lineHeight: 21 },
  clearButton: {
    position: "absolute",
    bottom: 28,
    alignSelf: "center",
    backgroundColor: BRAND.coral,
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 14,
    shadowColor: BRAND.coral,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  clearButtonText: { color: "#fff", fontSize: 16, fontWeight: "600" },
});
