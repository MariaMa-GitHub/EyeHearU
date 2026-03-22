import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";

const BRAND_TEAL = "#0D9488";

export default function RootLayout() {
  return (
    <>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: BRAND_TEAL },
          headerTintColor: "#fff",
          headerTitleStyle: { fontWeight: "bold" },
        }}
      >
        <Stack.Screen name="index" options={{ title: "Eye Hear U" }} />
        <Stack.Screen name="camera" options={{ title: "Translate" }} />
        <Stack.Screen name="history" options={{ title: "History" }} />
      </Stack>
    </>
  );
}
