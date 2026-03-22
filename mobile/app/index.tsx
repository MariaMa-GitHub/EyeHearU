import { useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
} from "react-native";
import { router } from "expo-router";

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

export default function HomeScreen() {
  const handAnim = useRef(new Animated.Value(0)).current;

  Animated.loop(
    Animated.sequence([
      Animated.timing(handAnim, { toValue: 1, duration: 1500, useNativeDriver: true }),
      Animated.timing(handAnim, { toValue: 0, duration: 1500, useNativeDriver: true }),
    ])
  ).start();

  const handRotate = handAnim.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: ["-8deg", "8deg", "-8deg"],
  });

  return (
    <View style={styles.container}>
      <View style={styles.hero}>
        <Animated.Text
          style={[styles.handIcon, { transform: [{ rotate: handRotate }] }]}
        >
          {"\u{1F91F}"}
        </Animated.Text>
        <Text style={styles.title}>
          Eye <Text style={styles.titleAccent}>Hear</Text> U
        </Text>
        <Text style={styles.subtitle}>
          ASL to English, one sign at a time
        </Text>
      </View>

      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => router.push("/camera")}
          activeOpacity={0.85}
        >
          <Text style={styles.primaryIcon}>{"\u{1F3A5}"}</Text>
          <Text style={styles.primaryButtonText}>Start Translating</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={() => router.push("/history")}
          activeOpacity={0.85}
        >
          <Text style={styles.secondaryIcon}>{"\u{1F4CB}"}</Text>
          <Text style={styles.secondaryButtonText}>View History</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.footer}>
        Record a 3-second video of an ASL sign{"\n"}to see the English translation
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: BRAND.bg,
    alignItems: "center",
    justifyContent: "center",
    padding: 28,
  },
  hero: { alignItems: "center", marginBottom: 44 },
  handIcon: { fontSize: 64, marginBottom: 12 },
  title: {
    fontSize: 40,
    fontWeight: "900",
    color: BRAND.textPrimary,
    letterSpacing: -1,
    marginBottom: 6,
  },
  titleAccent: { color: BRAND.coral },
  subtitle: { fontSize: 17, color: BRAND.textSecondary, textAlign: "center" },
  actions: { width: "100%", gap: 14, marginBottom: 28 },
  primaryButton: {
    backgroundColor: BRAND.teal,
    paddingVertical: 17,
    borderRadius: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    shadowColor: BRAND.teal,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 6,
  },
  primaryIcon: { fontSize: 20 },
  primaryButtonText: { color: "#fff", fontSize: 18, fontWeight: "700" },
  secondaryButton: {
    backgroundColor: BRAND.card,
    paddingVertical: 17,
    borderRadius: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    borderWidth: 1.5,
    borderColor: "#E2E8F0",
  },
  secondaryIcon: { fontSize: 20 },
  secondaryButtonText: { color: BRAND.teal, fontSize: 18, fontWeight: "600" },
  footer: {
    color: BRAND.textMuted,
    fontSize: 14,
    textAlign: "center",
    lineHeight: 20,
  },
});
