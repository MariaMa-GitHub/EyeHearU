/**
 * Tests for the Camera screen (app/camera.tsx).
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react-native";

/* ------------------------------------------------------------------ */
/*  Mocks                                                              */
/* ------------------------------------------------------------------ */

// Track the mock permission state so tests can control it
let mockPermission: { granted: boolean } | null = { granted: false };
const mockRequestPermission = jest.fn(async () => {
  mockPermission = { granted: true };
});

// Controllable recordAsync — by default resolves immediately; tests can override
let mockRecordAsync = jest.fn(async () => ({ uri: "file:///mock-video.mp4" }));
const mockStopRecording = jest.fn();

jest.mock("expo-camera", () => {
  const React = require("react");

  return {
    CameraView: React.forwardRef(
      (props: Record<string, unknown>, ref: React.Ref<unknown>) => {
        React.useImperativeHandle(ref, () => ({
          recordAsync: (...args: any[]) => mockRecordAsync(...args),
          stopRecording: (...args: any[]) => mockStopRecording(...args),
        }));
        const { View } = require("react-native");
        return <View {...props} testID="camera-view" />;
      }
    ),
    useCameraPermissions: () => [mockPermission, mockRequestPermission],
  };
});

jest.mock("expo-image-picker", () => ({
  launchImageLibraryAsync: jest.fn(async () => ({
    canceled: false,
    assets: [{ uri: "file:///picked-video.mp4" }],
  })),
}));

jest.mock("expo-speech", () => ({
  speak: jest.fn(),
}));

jest.mock("@react-native-async-storage/async-storage", () => ({
  __esModule: true,
  default: {
    getItem: jest.fn(async () => null),
    setItem: jest.fn(async () => {}),
    removeItem: jest.fn(async () => {}),
  },
}));

jest.mock("@expo/vector-icons", () => {
  const React = require("react");
  const { Text } = require("react-native");
  return {
    Ionicons: ({ name, ...props }: any) => <Text {...props}>{name}</Text>,
  };
});

jest.mock("../services/api", () => ({
  predictSign: jest.fn(async () => ({
    sign: "hello",
    confidence: 0.95,
    top_k: [{ sign: "hello", confidence: 0.95 }],
  })),
}));

import CameraScreen from "../app/camera";
import * as ImagePicker from "expo-image-picker";
import * as Speech from "expo-speech";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { predictSign } from "../services/api";
import { act, waitFor } from "@testing-library/react-native";

/* ------------------------------------------------------------------ */
/*  Tests                                                              */
/* ------------------------------------------------------------------ */
describe("CameraScreen", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Default: permission not granted
    mockPermission = { granted: false };
    // Reset to default instant-resolve behavior
    mockRecordAsync = jest.fn(async () => ({ uri: "file:///mock-video.mp4" }));
  });

  /* --- Permission screen --- */

  describe("permission request screen", () => {
    it("renders permission request when permission is not granted", () => {
      render(<CameraScreen />);
      expect(screen.getByText("Camera Access Needed")).toBeTruthy();
      expect(
        screen.getByText(
          "Eye Hear U uses the device camera to record ASL signs for translation."
        )
      ).toBeTruthy();
      expect(screen.getByText("Grant Permission")).toBeTruthy();
    });

    it("calls requestPermission when Grant Permission is pressed", () => {
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Grant Permission"));
      expect(mockRequestPermission).toHaveBeenCalledTimes(1);
    });

    it("renders empty view when permission is null (loading)", () => {
      mockPermission = null;
      const { toJSON } = render(<CameraScreen />);
      // Should render a plain View with no text children
      expect(screen.queryByText("Camera Access Needed")).toBeNull();
      expect(screen.queryByText("Record Sign")).toBeNull();
    });
  });

  /* --- Camera screen (permission granted) --- */

  describe("camera screen (permission granted)", () => {
    beforeEach(() => {
      mockPermission = { granted: true };
    });

    it("renders the camera view and record button", () => {
      render(<CameraScreen />);
      expect(screen.getByTestId("camera-view")).toBeTruthy();
      expect(screen.getByText("Record Sign")).toBeTruthy();
    });

    it("renders guidance text when idle", () => {
      render(<CameraScreen />);
      expect(
        screen.getByText(/Sign in front of the camera/)
      ).toBeTruthy();
    });

    it("renders the camera toggle button", () => {
      render(<CameraScreen />);
      expect(screen.getByText("camera-reverse-outline")).toBeTruthy();
    });

    it("renders the upload button", () => {
      render(<CameraScreen />);
      expect(screen.getByText("cloud-upload-outline")).toBeTruthy();
    });

    it("camera toggle changes facing prop", () => {
      render(<CameraScreen />);
      const cameraView = screen.getByTestId("camera-view");

      // Default should be "front"
      expect(cameraView.props.facing).toBe("front");

      // Tap toggle
      fireEvent.press(screen.getByText("camera-reverse-outline"));

      // Should now be "back"
      const updatedCameraView = screen.getByTestId("camera-view");
      expect(updatedCameraView.props.facing).toBe("back");
    });

    it("toggles back to front on second press", () => {
      render(<CameraScreen />);

      fireEvent.press(screen.getByText("camera-reverse-outline"));
      expect(screen.getByTestId("camera-view").props.facing).toBe("back");

      fireEvent.press(screen.getByText("camera-reverse-outline"));
      expect(screen.getByTestId("camera-view").props.facing).toBe("front");
    });

    it("upload button triggers image picker", async () => {
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("cloud-upload-outline"));

      expect(ImagePicker.launchImageLibraryAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          mediaTypes: ["videos"],
        })
      );
    });

    it("handles cancelled image picker gracefully", async () => {
      (ImagePicker.launchImageLibraryAsync as jest.Mock).mockResolvedValueOnce({
        canceled: true,
        assets: null,
      });

      render(<CameraScreen />);
      fireEvent.press(screen.getByText("cloud-upload-outline"));

      // Should not crash; should still show idle state
      // Wait a tick for promises to resolve
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(screen.getByText("Record Sign")).toBeTruthy();
    });

    /* --- recordAndPredict flow --- */

    it("records video and displays prediction result", async () => {
      (predictSign as jest.Mock).mockResolvedValueOnce({
        sign: "thank you",
        confidence: 0.9,
        top_k: [
          { sign: "thank you", confidence: 0.9 },
          { sign: "hello", confidence: 0.05 },
          { sign: "please", confidence: 0.03 },
        ],
      });

      render(<CameraScreen />);

      await act(async () => {
        fireEvent.press(screen.getByText("Record Sign"));
      });

      await waitFor(() => {
        expect(screen.getByText("thank you")).toBeTruthy();
      });

      // Verify confidence text
      expect(screen.getByText("90.0% confidence")).toBeTruthy();

      // Verify topK chips are rendered (items after the first)
      expect(screen.getByText("hello 5%")).toBeTruthy();
      expect(screen.getByText("please 3%")).toBeTruthy();

      // Verify prediction was saved to history
      expect(AsyncStorage.setItem).toHaveBeenCalled();

      // Verify auto-speak was called (confidence > 0.3)
      expect(Speech.speak).toHaveBeenCalledWith("thank you", {
        language: "en-US",
        rate: 0.9,
      });
    });

    it("shows error message when recordAndPredict fails", async () => {
      (predictSign as jest.Mock).mockRejectedValueOnce(
        new Error("Network timeout")
      );

      render(<CameraScreen />);

      await act(async () => {
        fireEvent.press(screen.getByText("Record Sign"));
      });

      await waitFor(() => {
        expect(screen.getByText("Request failed")).toBeTruthy();
      });
      expect(screen.getByText("Network timeout")).toBeTruthy();
    });

    /* --- pickAndPredict error handling --- */

    it("shows error when pickAndPredict throws", async () => {
      (ImagePicker.launchImageLibraryAsync as jest.Mock).mockRejectedValueOnce(
        new Error("Picker crashed")
      );

      render(<CameraScreen />);

      await act(async () => {
        fireEvent.press(screen.getByText("cloud-upload-outline"));
      });

      await waitFor(() => {
        expect(screen.getByText("Request failed")).toBeTruthy();
      });
      expect(screen.getByText("Picker crashed")).toBeTruthy();
    });

    /* --- stopRecording --- */

    it("shows stop button during recording and calls stopRecording", async () => {
      // Make recordAsync hang so isRecording stays true
      let resolveRecord!: (v: { uri: string }) => void;
      mockRecordAsync = jest.fn(
        () =>
          new Promise<{ uri: string }>((resolve) => {
            resolveRecord = resolve;
          })
      );

      render(<CameraScreen />);

      // Start recording — this won't resolve because recordAsync is pending
      act(() => {
        fireEvent.press(screen.getByText("Record Sign"));
      });

      // Now isRecording should be true, showing the stop button
      await waitFor(() => {
        expect(screen.getByText(/Stop/)).toBeTruthy();
      });

      // Also verify recording overlay is visible
      expect(
        screen.getByText(/Recording \d+s/)
      ).toBeTruthy();

      // Press stop
      fireEvent.press(screen.getByText(/Stop/));
      expect(mockStopRecording).toHaveBeenCalled();

      // Resolve the pending recordAsync to clean up
      await act(async () => {
        resolveRecord({ uri: "file:///mock-video.mp4" });
      });
    });

    /* --- speakPrediction --- */

    it("speaks prediction when Speak Again is pressed", async () => {
      (predictSign as jest.Mock).mockResolvedValueOnce({
        sign: "goodbye",
        confidence: 0.85,
        top_k: [{ sign: "goodbye", confidence: 0.85 }],
      });

      render(<CameraScreen />);

      await act(async () => {
        fireEvent.press(screen.getByText("Record Sign"));
      });

      await waitFor(() => {
        expect(screen.getByText("goodbye")).toBeTruthy();
      });

      // Clear the auto-speak call
      (Speech.speak as jest.Mock).mockClear();

      // Press "Speak Again"
      fireEvent.press(screen.getByText(/Speak Again/));

      expect(Speech.speak).toHaveBeenCalledWith("goodbye", {
        language: "en-US",
        rate: 0.9,
      });
    });

    /* --- saveToHistory with existing history --- */

    it("saves to history prepending to existing entries", async () => {
      const existingHistory = [
        {
          id: "old1",
          sign: "hi",
          confidence: 0.8,
          timestamp: "2025-01-01T00:00:00.000Z",
        },
      ];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValueOnce(
        JSON.stringify(existingHistory)
      );

      (predictSign as jest.Mock).mockResolvedValueOnce({
        sign: "yes",
        confidence: 0.92,
        top_k: [{ sign: "yes", confidence: 0.92 }],
      });

      render(<CameraScreen />);

      await act(async () => {
        fireEvent.press(screen.getByText("Record Sign"));
      });

      await waitFor(() => {
        expect(AsyncStorage.setItem).toHaveBeenCalled();
      });

      const savedData = JSON.parse(
        (AsyncStorage.setItem as jest.Mock).mock.calls[0][1]
      );
      expect(savedData.length).toBe(2);
      expect(savedData[0].sign).toBe("yes");
      expect(savedData[1].sign).toBe("hi");
    });

    /* --- pulse animation runs when recording --- */

    it("shows recording overlay with pulse animation when recording", async () => {
      // Make recordAsync hang so isRecording stays true and pulse anim runs
      let resolveRecord!: (v: { uri: string }) => void;
      mockRecordAsync = jest.fn(
        () =>
          new Promise<{ uri: string }>((resolve) => {
            resolveRecord = resolve;
          })
      );

      jest.useFakeTimers();

      render(<CameraScreen />);

      act(() => {
        fireEvent.press(screen.getByText("Record Sign"));
      });

      // isRecording is true, the pulse useEffect should have started
      await waitFor(() => {
        expect(
          screen.getByText(/Recording \d+s/)
        ).toBeTruthy();
      });

      // Advance timers to exercise the Animated.loop (pulse animation)
      act(() => {
        jest.advanceTimersByTime(1500);
      });

      // Still recording
      expect(
        screen.getByText(/Recording \d+s/)
      ).toBeTruthy();

      // Cleanup: resolve and restore timers
      jest.useRealTimers();
      await act(async () => {
        resolveRecord({ uri: "file:///mock-video.mp4" });
      });
    });

    it("countdown hits final tick and clears interval at 0s label", async () => {
      let resolveRecord!: (v: { uri: string }) => void;
      mockRecordAsync = jest.fn(
        () =>
          new Promise<{ uri: string }>((resolve) => {
            resolveRecord = resolve;
          })
      );

      jest.useFakeTimers({ legacyFakeTimers: false });

      render(<CameraScreen />);

      act(() => {
        fireEvent.press(screen.getByText("Record Sign"));
      });

      await waitFor(() => {
        expect(screen.getByText("Recording 5s")).toBeTruthy();
      });

      // 5s → 4 → 3 → 2 → 1 → (final tick clears interval, shows "Recording" only)
      act(() => {
        jest.advanceTimersByTime(5000);
      });

      await waitFor(() => {
        expect(screen.getByText("Recording")).toBeTruthy();
      });
      expect(screen.queryByText(/Recording \d+s/)).toBeNull();

      jest.useRealTimers();
      await act(async () => {
        resolveRecord({ uri: "file:///mock-video.mp4" });
      });
    });

    /* --- upload flow with successful prediction --- */

    it("displays prediction after successful upload", async () => {
      (ImagePicker.launchImageLibraryAsync as jest.Mock).mockResolvedValueOnce({
        canceled: false,
        assets: [{ uri: "file:///picked-video.mp4" }],
      });
      (predictSign as jest.Mock).mockResolvedValueOnce({
        sign: "water",
        confidence: 0.78,
        top_k: [
          { sign: "water", confidence: 0.78 },
          { sign: "drink", confidence: 0.12 },
        ],
      });

      render(<CameraScreen />);

      await act(async () => {
        fireEvent.press(screen.getByText("cloud-upload-outline"));
      });

      await waitFor(() => {
        expect(screen.getByText("water")).toBeTruthy();
      });

      expect(screen.getByText("78.0% confidence")).toBeTruthy();
      expect(screen.getByText("drink 12%")).toBeTruthy();
    });
  });
});
