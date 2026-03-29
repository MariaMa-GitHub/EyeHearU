/**
 * Tests for the Camera screen (app/camera.tsx).
 */

import React from "react";
import { Platform, StyleSheet } from "react-native";
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
let mockRecordAsync: jest.Mock<
  Promise<{ uri?: string }>,
  []
> = jest.fn(async () => ({ uri: "file:///mock-video.mp4" }));
const mockStopRecording = jest.fn();

jest.mock("expo-camera", () => {
  const React = require("react");

  return {
    CameraView: React.forwardRef(
      (props: Record<string, unknown>, ref: React.Ref<unknown>) => {
        React.useImperativeHandle(ref, () => ({
          recordAsync: mockRecordAsync,
          stopRecording: mockStopRecording,
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

const expoVideoTest = {
  statusListener: null as null | ((payload: { status: string }) => void),
};

jest.mock("expo-video", () => {
  const React = require("react");
  const { View } = require("react-native");
  return {
    useVideoPlayer: jest.fn(
      (_url: string | null, setup?: (p: Record<string, unknown>) => void) => {
        const p = {
          loop: false,
          play: jest.fn(),
          addListener: jest.fn(
            (event: string, cb: (payload: { status: string }) => void) => {
              if (event === "statusChange") {
                expoVideoTest.statusListener = cb;
              }
              return { remove: jest.fn() };
            }
          ),
        };
        if (setup) setup(p);
        return p;
      }
    ),
    VideoView: (props: Record<string, unknown>) => (
      <View testID="expo-video-view" {...props} />
    ),
  };
});

jest.mock("expo-web-browser", () => ({
  openBrowserAsync: jest.fn(async () => ({ type: "dismiss" })),
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
  predictSentence: jest.fn(async () => ({
    clips: [],
    beam: [{ glosses: ["hello"], score: 1, english: "Hello there." }],
    best_glosses: ["hello"],
    english: "Hello there.",
  })),
}));

import CameraScreen from "../app/camera";
import * as ImagePicker from "expo-image-picker";
import * as Speech from "expo-speech";
import * as WebBrowser from "expo-web-browser";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { predictSign, predictSentence } from "../services/api";
import { act, waitFor } from "@testing-library/react-native";

/* ------------------------------------------------------------------ */
/*  Tests                                                              */
/* ------------------------------------------------------------------ */
describe("CameraScreen", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    expoVideoTest.statusListener = null;
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

    describe("platform-specific layout", () => {
      const setPlatformOS = (os: "ios" | "android") => {
        Object.defineProperty(Platform, "OS", {
          configurable: true,
          enumerable: true,
          value: os,
          writable: true,
        });
      };

      afterEach(() => {
        setPlatformOS("ios");
      });

      it("uses iOS top inset for controls and recording overlay", () => {
        setPlatformOS("ios");
        render(<CameraScreen />);
        const row = screen.getByTestId("camera-top-controls");
        expect(StyleSheet.flatten(row.props.style).top).toBe(16);
      });

      it("uses smaller top inset on Android", () => {
        setPlatformOS("android");
        render(<CameraScreen />);
        const row = screen.getByTestId("camera-top-controls");
        expect(StyleSheet.flatten(row.props.style).top).toBe(12);
      });

      it("positions the mode row for the current platform", () => {
        setPlatformOS("ios");
        const { unmount } = render(<CameraScreen />);
        expect(
          StyleSheet.flatten(screen.getByTestId("camera-mode-row").props.style)
            .top
        ).toBe(64);
        unmount();
        setPlatformOS("android");
        render(<CameraScreen />);
        expect(
          StyleSheet.flatten(screen.getByTestId("camera-mode-row").props.style)
            .top
        ).toBe(58);
      });

      it("applies platform top inset on the recording overlay while recording", async () => {
        setPlatformOS("android");
        let resolveRecord!: (v: { uri: string }) => void;
        mockRecordAsync = jest.fn(
          () =>
            new Promise<{ uri: string }>((resolve) => {
              resolveRecord = resolve;
            })
        );

        render(<CameraScreen />);

        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });

        await waitFor(() => {
          expect(screen.getByTestId("camera-recording-overlay")).toBeTruthy();
        });

        const overlay = screen.getByTestId("camera-recording-overlay");
        expect(StyleSheet.flatten(overlay.props.style).top).toBe(12);

        await act(async () => {
          resolveRecord({ uri: "file:///mock-video.mp4" });
        });
      });
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

    it("does not auto-speak when confidence is not above 0.3", async () => {
      (predictSign as jest.Mock).mockResolvedValueOnce({
        sign: "quiet",
        confidence: 0.3,
        top_k: [{ sign: "quiet", confidence: 0.3 }],
      });
      render(<CameraScreen />);
      await act(async () => {
        fireEvent.press(screen.getByText("Record Sign"));
      });
      await waitFor(() => {
        expect(screen.getByText("quiet")).toBeTruthy();
      });
      expect(Speech.speak).not.toHaveBeenCalled();
    });

    it("skips history and speech when the predicted sign is empty", async () => {
      (predictSign as jest.Mock).mockResolvedValueOnce({
        sign: "",
        confidence: 0.95,
        top_k: [],
      });
      render(<CameraScreen />);
      await act(async () => {
        fireEvent.press(screen.getByText("Record Sign"));
      });
      await waitFor(() => {
        expect(
          screen.getByText(/Sign in front of the camera/)
        ).toBeTruthy();
      });
      expect(AsyncStorage.setItem).not.toHaveBeenCalled();
      expect(Speech.speak).not.toHaveBeenCalled();
    });

    it("returns early when recording yields no file URI", async () => {
      mockRecordAsync = jest.fn(async () => ({}));
      render(<CameraScreen />);
      await act(async () => {
        fireEvent.press(screen.getByText("Record Sign"));
      });
      await waitFor(() => {
        expect(mockRecordAsync).toHaveBeenCalled();
      });
      expect(screen.getByText("Record Sign")).toBeTruthy();
      expect(predictSign).not.toHaveBeenCalled();
    });

    it("suppresses single-sign history persistence errors", async () => {
      (AsyncStorage.setItem as jest.Mock).mockRejectedValueOnce(
        new Error("disk full")
      );
      (predictSign as jest.Mock).mockResolvedValueOnce({
        sign: "persist",
        confidence: 0.91,
        top_k: [{ sign: "persist", confidence: 0.91 }],
      });
      render(<CameraScreen />);
      await act(async () => {
        fireEvent.press(screen.getByText("Record Sign"));
      });
      await waitFor(() => {
        expect(screen.getByText("persist")).toBeTruthy();
      });
    });

    it("defaults missing beam scores to 0 when saving sentence history", async () => {
      (predictSentence as jest.Mock).mockResolvedValueOnce({
        clips: [],
        beam: [{ glosses: ["solo"] }] as {
          glosses: string[];
          score?: number;
          english: string;
        }[],
        best_glosses: ["solo"],
        english: "Solo.",
      });
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));
      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });
      await waitFor(() => {
        expect(screen.getByText("Translate")).toBeTruthy();
      });
      await act(async () => {
        fireEvent.press(screen.getByText("Translate"));
      });
      await waitFor(() => {
        expect(screen.getByText("Solo.")).toBeTruthy();
      });
      expect(AsyncStorage.setItem).toHaveBeenCalled();
    });

    it("handles sentence results with empty english", async () => {
      (predictSentence as jest.Mock).mockResolvedValueOnce({
        clips: [],
        beam: [],
        best_glosses: [],
        english: "",
      });
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));
      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });
      await waitFor(() => {
        expect(screen.getByText("Translate")).toBeTruthy();
      });
      await act(async () => {
        fireEvent.press(screen.getByText("Translate"));
      });
      await waitFor(() => {
        expect(screen.getByText("Sentence")).toBeTruthy();
      });
      expect(Speech.speak).not.toHaveBeenCalled();
    });

    it("suppresses sentence history read errors", async () => {
      (AsyncStorage.getItem as jest.Mock).mockRejectedValueOnce(
        new Error("cannot read history")
      );
      (predictSentence as jest.Mock).mockResolvedValueOnce({
        clips: [],
        beam: [{ glosses: ["y"], score: 0.5, english: "Y." }],
        best_glosses: ["y"],
        english: "Y.",
      });
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));
      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });
      await waitFor(() => {
        expect(screen.getByText("Translate")).toBeTruthy();
      });
      await act(async () => {
        fireEvent.press(screen.getByText("Translate"));
      });
      await waitFor(() => {
        expect(screen.getByText("Y.")).toBeTruthy();
      });
    });

    it("suppresses sentence history persistence errors", async () => {
      (predictSentence as jest.Mock).mockResolvedValueOnce({
        clips: [],
        beam: [{ glosses: ["x"], score: 1, english: "X." }],
        best_glosses: ["x"],
        english: "X.",
      });
      (AsyncStorage.setItem as jest.Mock).mockRejectedValueOnce(
        new Error("storage error")
      );
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));
      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });
      await waitFor(() => {
        expect(screen.getByText("Translate")).toBeTruthy();
      });
      await act(async () => {
        fireEvent.press(screen.getByText("Translate"));
      });
      await waitFor(() => {
        expect(screen.getByText("X.")).toBeTruthy();
      });
    });

    it("shows ellipsis on sentence buttons while Translate is in flight", async () => {
      let resolveSentence!: (v: {
        clips: unknown[];
        beam: { glosses: string[]; score: number; english: string }[];
        best_glosses: string[];
        english: string;
      }) => void;
      (predictSentence as jest.Mock).mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveSentence = resolve;
          })
      );
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));
      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });
      await waitFor(() => {
        expect(screen.getByText("Translate")).toBeTruthy();
      });
      await act(async () => {
        fireEvent.press(screen.getByText("Translate"));
      });
      await waitFor(() => {
        expect(screen.getAllByText("…").length).toBeGreaterThanOrEqual(1);
      });
      await act(async () => {
        resolveSentence({
          clips: [],
          beam: [{ glosses: ["a"], score: 1, english: "A." }],
          best_glosses: ["a"],
          english: "A.",
        });
      });
    });

    it("shows stop label on Add sign while recording in Multi-sign", async () => {
      let resolveRecord!: (v: { uri: string }) => void;
      mockRecordAsync = jest.fn(
        () =>
          new Promise<{ uri: string }>((resolve) => {
            resolveRecord = resolve;
          })
      );
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));
      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });
      await waitFor(() => {
        expect(screen.getByText(/Stop/)).toBeTruthy();
      });
      await act(async () => {
        resolveRecord({ uri: "file:///mock-video.mp4" });
      });
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

    it("treats missing top_k as an empty list", async () => {
      (predictSign as jest.Mock).mockResolvedValueOnce({
        sign: "only",
        confidence: 0.91,
      });

      render(<CameraScreen />);

      await act(async () => {
        fireEvent.press(screen.getByText("Record Sign"));
      });

      await waitFor(() => {
        expect(screen.getByText("only")).toBeTruthy();
      });
      expect(screen.queryByText("play-circle-outline")).toBeNull();
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

    it("shows non-Error rejection message when recordAndPredict fails", async () => {
      (predictSign as jest.Mock).mockRejectedValueOnce("plain string failure");

      render(<CameraScreen />);

      await act(async () => {
        fireEvent.press(screen.getByText("Record Sign"));
      });

      await waitFor(() => {
        expect(screen.getByText("Request failed")).toBeTruthy();
      });
      expect(screen.getByText("plain string failure")).toBeTruthy();
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

    it("shows non-Error message when pickAndPredict throws a string", async () => {
      (ImagePicker.launchImageLibraryAsync as jest.Mock).mockRejectedValueOnce(
        "picker string error"
      );

      render(<CameraScreen />);

      await act(async () => {
        fireEvent.press(screen.getByText("cloud-upload-outline"));
      });

      await waitFor(() => {
        expect(screen.getByText("Request failed")).toBeTruthy();
      });
      expect(screen.getByText("picker string error")).toBeTruthy();
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

    it("caps persisted sentence history at 100 entries", async () => {
      const many = Array.from({ length: 101 }, (_, i) => ({
        id: `id-${i}`,
        sign: "old",
        confidence: 0.5,
        timestamp: "2025-01-01T00:00:00.000Z",
      }));
      (AsyncStorage.getItem as jest.Mock).mockResolvedValueOnce(
        JSON.stringify(many)
      );
      (predictSentence as jest.Mock).mockResolvedValueOnce({
        clips: [],
        beam: [{ glosses: ["new"], score: 2, english: "New sentence." }],
        best_glosses: ["new"],
        english: "New sentence.",
      });
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));
      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });
      await waitFor(() => {
        expect(screen.getByText("Translate")).toBeTruthy();
      });
      await act(async () => {
        fireEvent.press(screen.getByText("Translate"));
      });
      await waitFor(() => {
        expect(screen.getByText("New sentence.")).toBeTruthy();
      });
      const savedData = JSON.parse(
        (AsyncStorage.setItem as jest.Mock).mock.calls[0][1]
      );
      expect(savedData.length).toBe(100);
      expect(savedData[0].sign).toBe("New sentence.");
    });

    it("caps persisted history at 100 entries", async () => {
      const many = Array.from({ length: 101 }, (_, i) => ({
        id: `id-${i}`,
        sign: "old",
        confidence: 0.5,
        timestamp: "2025-01-01T00:00:00.000Z",
      }));
      (AsyncStorage.getItem as jest.Mock).mockResolvedValueOnce(
        JSON.stringify(many)
      );
      (predictSign as jest.Mock).mockResolvedValueOnce({
        sign: "newest",
        confidence: 0.99,
        top_k: [{ sign: "newest", confidence: 0.99 }],
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
      expect(savedData.length).toBe(100);
      expect(savedData[0].sign).toBe("newest");
    });

    it("ignores history persistence errors without breaking the flow", async () => {
      (AsyncStorage.getItem as jest.Mock).mockResolvedValueOnce(null);
      (AsyncStorage.setItem as jest.Mock).mockRejectedValueOnce(
        new Error("storage full")
      );
      (predictSign as jest.Mock).mockResolvedValueOnce({
        sign: "ok",
        confidence: 0.95,
        top_k: [{ sign: "ok", confidence: 0.95 }],
      });

      render(<CameraScreen />);

      await act(async () => {
        fireEvent.press(screen.getByText("Record Sign"));
      });

      await waitFor(() => {
        expect(screen.getByText("ok")).toBeTruthy();
      });
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

    it("counts recording countdown to zero while still recording", async () => {
      let resolveRecord!: (v: { uri: string }) => void;
      mockRecordAsync = jest.fn(
        () =>
          new Promise<{ uri: string }>((resolve) => {
            resolveRecord = resolve;
          })
      );

      jest.useFakeTimers({ advanceTimers: true });

      render(<CameraScreen />);

      act(() => {
        fireEvent.press(screen.getByText("Record Sign"));
      });

      await waitFor(() => {
        expect(screen.getByText(/Recording 5s/)).toBeTruthy();
      });

      for (let s = 4; s >= 1; s--) {
        act(() => {
          jest.advanceTimersByTime(1000);
        });
        expect(screen.getByText(new RegExp(`Recording ${s}s`))).toBeTruthy();
      }

      act(() => {
        jest.advanceTimersByTime(1000);
      });
      expect(screen.getByText(/Recording(?! \d)/)).toBeTruthy();

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

    /* --- Multi-sign (sentence) mode --- */

    it("switches to Multi-sign and does not call predictSign when recording a clip", async () => {
      render(<CameraScreen />);

      fireEvent.press(screen.getByText("Multi-sign"));

      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });

      await waitFor(() => {
        expect(screen.getByText(/1 clip\(s\)/)).toBeTruthy();
      });

      expect(predictSign).not.toHaveBeenCalled();
      expect(predictSentence).not.toHaveBeenCalled();
    });

    it("appends a gallery video to sentence clips in Multi-sign without predictSign", async () => {
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));

      await act(async () => {
        fireEvent.press(screen.getByText("cloud-upload-outline"));
      });

      await waitFor(() => {
        expect(screen.getByText(/1 clip\(s\)/)).toBeTruthy();
      });

      expect(predictSign).not.toHaveBeenCalled();
      expect(predictSentence).not.toHaveBeenCalled();
    });

    it("switches back to Single sign mode", () => {
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));
      fireEvent.press(screen.getByText("Single sign"));
      expect(screen.getByText("Record Sign")).toBeTruthy();
    });

    it("calls predictSentence and shows sentence result when Translate is pressed", async () => {
      (predictSentence as jest.Mock).mockResolvedValueOnce({
        clips: [{ top_k: [{ sign: "yes", confidence: 0.9 }] }],
        beam: [{ glosses: ["yes"], score: 2, english: "Yes please." }],
        best_glosses: ["yes"],
        english: "Yes please.",
      });

      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));

      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });

      await waitFor(() => {
        expect(screen.getByText("Translate")).toBeTruthy();
      });

      await act(async () => {
        fireEvent.press(screen.getByText("Translate"));
      });

      await waitFor(() => {
        expect(screen.getByText("Yes please.")).toBeTruthy();
      });

      expect(predictSentence).toHaveBeenCalledWith(["file:///mock-video.mp4"]);
      expect(Speech.speak).toHaveBeenCalledWith("Yes please.", {
        language: "en-US",
        rate: 0.9,
      });
    });

    it("clears sentence clips and result when Clear clips is pressed", async () => {
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));

      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });

      await waitFor(() => {
        expect(screen.getByText(/1 clip\(s\)/)).toBeTruthy();
      });

      fireEvent.press(screen.getByText("Clear clips"));

      await waitFor(() => {
        expect(screen.queryByText(/1 clip\(s\)/)).toBeNull();
      });
    });

    it("shows New sentence button after translation and clears state when pressed", async () => {
      (predictSentence as jest.Mock).mockResolvedValueOnce({
        clips: [],
        beam: [{ glosses: ["hello"], score: 1, english: "Hello." }],
        best_glosses: ["hello"],
        english: "Hello.",
      });

      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));

      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });

      await waitFor(() => {
        expect(screen.getByText("Translate")).toBeTruthy();
      });

      await act(async () => {
        fireEvent.press(screen.getByText("Translate"));
      });

      await waitFor(() => {
        expect(screen.getByText("Hello.")).toBeTruthy();
      });

      expect(screen.getByText("New sentence")).toBeTruthy();

      fireEvent.press(screen.getByText("New sentence"));

      await waitFor(() => {
        expect(screen.queryByText("Hello.")).toBeNull();
        expect(screen.queryByText("New sentence")).toBeNull();
      });
    });

    it("shows a string message when predictSentence rejects with a non-Error", async () => {
      (predictSentence as jest.Mock).mockRejectedValueOnce("sentence unavailable");
      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));
      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });
      await waitFor(() => {
        expect(screen.getByText("Translate")).toBeTruthy();
      });
      await act(async () => {
        fireEvent.press(screen.getByText("Translate"));
      });
      await waitFor(() => {
        expect(screen.getByText("Request failed")).toBeTruthy();
      });
      expect(screen.getByText("sentence unavailable")).toBeTruthy();
    });

    it("shows error when predictSentence fails", async () => {
      (predictSentence as jest.Mock).mockRejectedValueOnce(
        new Error("Sentence API down")
      );

      render(<CameraScreen />);
      fireEvent.press(screen.getByText("Multi-sign"));

      await act(async () => {
        fireEvent.press(screen.getByText("Add sign"));
      });

      await waitFor(() => {
        expect(screen.getByText("Translate")).toBeTruthy();
      });

      await act(async () => {
        fireEvent.press(screen.getByText("Translate"));
      });

      await waitFor(() => {
        expect(screen.getByText("Request failed")).toBeTruthy();
      });
      expect(screen.getByText("Sentence API down")).toBeTruthy();
    });

    describe("SignASL video modal", () => {
      let fetchSpy: jest.SpiedFunction<typeof fetch>;

      beforeEach(() => {
        fetchSpy = jest.spyOn(global, "fetch").mockResolvedValue(
          new Response(
            '<video src="https://media.signbsl.com/hello.mp4"></video>',
            { status: 200 }
          )
        );
      });

      afterEach(() => {
        fetchSpy.mockRestore();
      });

      it("opens reference video modal, plays when ready, then closes", async () => {
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "hello",
          confidence: 0.9,
          top_k: [{ sign: "hello", confidence: 0.9 }],
        });
        render(<CameraScreen />);
        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("hello")).toBeTruthy();
        });
        await act(async () => {
          fireEvent.press(screen.getByText("hello"));
        });
        await waitFor(() => {
          expect(screen.getByTestId("sign-video-modal").props.visible).toBe(
            true
          );
        });
        await act(async () => {
          expoVideoTest.statusListener?.({ status: "readyToPlay" });
        });
        expect(screen.getByTestId("expo-video-view")).toBeTruthy();
        await act(async () => {
          fireEvent.press(screen.getByText("close-circle"));
        });
        await waitFor(() => {
          expect(screen.queryByTestId("sign-video-modal")).toBeNull();
        });
      });

      it("falls back after player errors and opens SignASL in the browser", async () => {
        fetchSpy.mockResolvedValueOnce(
          new Response("x https://first.mp4 y https://second.mp4 z", {
            status: 200,
          })
        );
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "help",
          confidence: 0.88,
          top_k: [{ sign: "help", confidence: 0.88 }],
        });
        render(<CameraScreen />);
        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("help")).toBeTruthy();
        });
        await act(async () => {
          fireEvent.press(screen.getByText("help"));
        });
        await waitFor(() => {
          expect(screen.getByTestId("sign-video-modal").props.visible).toBe(
            true
          );
        });
        await act(async () => {
          expoVideoTest.statusListener?.({ status: "error" });
        });
        await act(async () => {
          expoVideoTest.statusListener?.({ status: "error" });
        });
        await waitFor(() => {
          expect(screen.getByText("Video unavailable")).toBeTruthy();
        });
        (WebBrowser.openBrowserAsync as jest.Mock).mockClear();
        await act(async () => {
          fireEvent.press(screen.getByText("Open in Browser"));
        });
        expect(WebBrowser.openBrowserAsync).toHaveBeenCalledWith(
          "https://www.signasl.org/sign/help"
        );
        await act(async () => {
          fireEvent.press(screen.getByText("More videos on SignASL.org"));
        });
        expect(WebBrowser.openBrowserAsync).toHaveBeenCalledTimes(2);
      });

      it("deduplicates repeated mp4 URLs from the sign page", async () => {
        fetchSpy.mockResolvedValueOnce(
          new Response(
            "https://same.mp4 repeated https://same.mp4",
            { status: 200 }
          )
        );
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "dedup",
          confidence: 0.8,
          top_k: [{ sign: "dedup", confidence: 0.8 }],
        });
        render(<CameraScreen />);
        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("dedup")).toBeTruthy();
        });
        await act(async () => {
          fireEvent.press(screen.getByText("dedup"));
        });
        await waitFor(() => {
          expect(screen.getByTestId("sign-video-modal").props.visible).toBe(
            true
          );
        });
        await act(async () => {
          expoVideoTest.statusListener?.({ status: "error" });
        });
        await waitFor(() => {
          expect(screen.getByText("Video unavailable")).toBeTruthy();
        });
      });

      it("shows unavailable when the sign page has no mp4 URLs", async () => {
        fetchSpy.mockResolvedValueOnce(
          new Response("<html>No videos</html>", { status: 200 })
        );
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "empty",
          confidence: 0.7,
          top_k: [{ sign: "empty", confidence: 0.7 }],
        });
        render(<CameraScreen />);
        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("empty")).toBeTruthy();
        });
        await act(async () => {
          fireEvent.press(screen.getByText("empty"));
        });
        await waitFor(() => {
          expect(screen.getByText("Video unavailable")).toBeTruthy();
        });
      });

      it("shows unavailable when the sign page returns a non-OK status", async () => {
        fetchSpy.mockResolvedValueOnce(new Response("", { status: 500 }));
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "oops",
          confidence: 0.7,
          top_k: [{ sign: "oops", confidence: 0.7 }],
        });
        render(<CameraScreen />);
        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("oops")).toBeTruthy();
        });
        await act(async () => {
          fireEvent.press(screen.getByText("oops"));
        });
        await waitFor(() => {
          expect(screen.getByText("Video unavailable")).toBeTruthy();
        });
      });

      it("shows unavailable when fetch rejects", async () => {
        fetchSpy.mockRejectedValueOnce(new Error("offline"));
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "bye",
          confidence: 0.7,
          top_k: [{ sign: "bye", confidence: 0.7 }],
        });
        render(<CameraScreen />);
        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("bye")).toBeTruthy();
        });
        await act(async () => {
          fireEvent.press(screen.getByText("bye"));
        });
        await waitFor(() => {
          expect(screen.getByText("Video unavailable")).toBeTruthy();
        });
      });

      it("opens video from a top-k chip with a different sign", async () => {
        fetchSpy.mockResolvedValueOnce(
          new Response('<a href="https://other.mp4">', { status: 200 })
        );
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "mom",
          confidence: 0.6,
          top_k: [
            { sign: "mom", confidence: 0.6 },
            { sign: "mother", confidence: 0.3 },
          ],
        });
        render(<CameraScreen />);
        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("mother 30%")).toBeTruthy();
        });
        await act(async () => {
          fireEvent.press(screen.getByText("mother 30%"));
        });
        await waitFor(() => {
          expect(global.fetch).toHaveBeenCalledWith(
            "https://www.signasl.org/sign/mother"
          );
        });
      });

      it("shows unavailable when the gloss slug is empty and skips browser when slug stays empty", async () => {
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "###",
          confidence: 0.5,
          top_k: [{ sign: "###", confidence: 0.5 }],
        });
        render(<CameraScreen />);
        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("###")).toBeTruthy();
        });
        await act(async () => {
          fireEvent.press(screen.getByText("###"));
        });
        await waitFor(() => {
          expect(screen.getByText("Video unavailable")).toBeTruthy();
        });
        expect(fetchSpy).not.toHaveBeenCalled();
        (WebBrowser.openBrowserAsync as jest.Mock).mockClear();
        await act(async () => {
          fireEvent.press(screen.getByText("More videos on SignASL.org"));
        });
        expect(WebBrowser.openBrowserAsync).not.toHaveBeenCalled();
      });
    });
  });
});
