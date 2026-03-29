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
}));

import CameraScreen from "../app/camera";
import * as ImagePicker from "expo-image-picker";
import * as Speech from "expo-speech";
import * as WebBrowser from "expo-web-browser";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { predictSign } from "../services/api";
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

    /* --- ASL video dictionary modal --- */

    describe("video dictionary modal", () => {
      it("opens modal when tapping the detected sign and shows loading state", async () => {
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "hello",
          confidence: 0.95,
          top_k: [{ sign: "hello", confidence: 0.95 }],
        });

        render(<CameraScreen />);

        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });

        await waitFor(() => {
          expect(screen.getByText("hello")).toBeTruthy();
        });

        fireEvent.press(screen.getByText("hello"));

        await waitFor(() => {
          expect(screen.getByTestId("sign-video-modal")).toBeTruthy();
        });
        expect(screen.getByText("Loading video...")).toBeTruthy();
        expect(screen.getByTestId("expo-video-view")).toBeTruthy();
      });

      it("hides loading overlay when video reports readyToPlay", async () => {
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "water",
          confidence: 0.8,
          top_k: [{ sign: "water", confidence: 0.8 }],
        });

        render(<CameraScreen />);

        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("water")).toBeTruthy();
        });

        fireEvent.press(screen.getByText("water"));

        await waitFor(() => {
          expect(expoVideoTest.statusListener).not.toBeNull();
        });

        await act(async () => {
          expoVideoTest.statusListener!({ status: "readyToPlay" });
        });

        expect(screen.queryByText("Loading video...")).toBeNull();
      });

      it("advances through fallback URLs on error and shows fallback UI after last source", async () => {
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "thanks",
          confidence: 0.88,
          top_k: [{ sign: "thanks", confidence: 0.88 }],
        });

        render(<CameraScreen />);

        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("thanks")).toBeTruthy();
        });

        fireEvent.press(screen.getByText("thanks"));

        await waitFor(() => {
          expect(expoVideoTest.statusListener).not.toBeNull();
        });

        for (let i = 0; i < 4; i++) {
          await act(async () => {
            expoVideoTest.statusListener!({ status: "error" });
          });
        }

        await waitFor(() => {
          expect(screen.getByText("Video unavailable")).toBeTruthy();
        });
        expect(screen.getByText("Open in Browser")).toBeTruthy();
      });

      it("opens SignASL in browser from error fallback", async () => {
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "book",
          confidence: 0.77,
          top_k: [{ sign: "book", confidence: 0.77 }],
        });

        render(<CameraScreen />);

        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("book")).toBeTruthy();
        });

        fireEvent.press(screen.getByText("book"));

        await waitFor(() => {
          expect(expoVideoTest.statusListener).not.toBeNull();
        });

        for (let i = 0; i < 4; i++) {
          await act(async () => {
            expoVideoTest.statusListener!({ status: "error" });
          });
        }

        await waitFor(() => {
          expect(screen.getByText("Video unavailable")).toBeTruthy();
        });

        await act(async () => {
          fireEvent.press(screen.getByText("Open in Browser"));
        });

        expect(WebBrowser.openBrowserAsync).toHaveBeenCalledWith(
          "https://www.signasl.org/sign/book"
        );
      });

      it("opens SignASL from footer link when video loads", async () => {
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "book",
          confidence: 0.77,
          top_k: [{ sign: "book", confidence: 0.77 }],
        });

        render(<CameraScreen />);

        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("book")).toBeTruthy();
        });

        fireEvent.press(screen.getByText("book"));

        await waitFor(() => {
          expect(expoVideoTest.statusListener).not.toBeNull();
        });

        await act(async () => {
          expoVideoTest.statusListener!({ status: "readyToPlay" });
        });

        await act(async () => {
          fireEvent.press(screen.getByText("More videos on SignASL.org"));
        });

        expect(WebBrowser.openBrowserAsync).toHaveBeenCalledWith(
          "https://www.signasl.org/sign/book"
        );
      });

      it("closes the modal via close button and onRequestClose", async () => {
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "yes",
          confidence: 0.9,
          top_k: [{ sign: "yes", confidence: 0.9 }],
        });

        render(<CameraScreen />);

        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("yes")).toBeTruthy();
        });

        fireEvent.press(screen.getByText("yes"));

        await waitFor(() => {
          expect(screen.getByTestId("sign-video-modal")).toBeTruthy();
        });

        fireEvent.press(screen.getByText("close-circle"));

        await waitFor(() => {
          expect(screen.queryByTestId("sign-video-modal")).toBeNull();
        });

        fireEvent.press(screen.getByText("yes"));
        await waitFor(() => {
          expect(screen.getByTestId("sign-video-modal")).toBeTruthy();
        });

        const modal = screen.getByTestId("sign-video-modal");
        fireEvent(modal, "onRequestClose");

        await waitFor(() => {
          expect(screen.queryByTestId("sign-video-modal")).toBeNull();
        });
      });

      it("opens the video modal from a top-K chip", async () => {
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "a",
          confidence: 0.5,
          top_k: [
            { sign: "a", confidence: 0.5 },
            { sign: "b", confidence: 0.3 },
          ],
        });

        render(<CameraScreen />);

        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });
        await waitFor(() => {
          expect(screen.getByText("b 30%")).toBeTruthy();
        });

        fireEvent.press(screen.getByText("b 30%"));

        await waitFor(() => {
          expect(screen.getByText("b")).toBeTruthy();
          expect(screen.getByTestId("sign-video-modal")).toBeTruthy();
        });
      });

      it("returns early when recordAsync yields no uri", async () => {
        mockRecordAsync = jest.fn(async () => ({}));

        render(<CameraScreen />);

        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });

        await waitFor(() => {
          expect(mockRecordAsync).toHaveBeenCalled();
        });

        expect(predictSign).not.toHaveBeenCalled();
        expect(screen.getByText("Record Sign")).toBeTruthy();
      });

      it("does not auto-speak when confidence is at or below 0.3", async () => {
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "maybe",
          confidence: 0.25,
          top_k: [{ sign: "maybe", confidence: 0.25 }],
        });

        render(<CameraScreen />);

        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });

        await waitFor(() => {
          expect(screen.getByText("maybe")).toBeTruthy();
        });

        expect(Speech.speak).not.toHaveBeenCalled();
      });

      it("skips history and speech when sign is empty", async () => {
        (predictSign as jest.Mock).mockResolvedValueOnce({
          sign: "",
          confidence: 0.99,
          top_k: [],
        });

        render(<CameraScreen />);

        await act(async () => {
          fireEvent.press(screen.getByText("Record Sign"));
        });

        await waitFor(() => {
          expect(predictSign).toHaveBeenCalled();
        });

        expect(AsyncStorage.setItem).not.toHaveBeenCalled();
        expect(Speech.speak).not.toHaveBeenCalled();
      });
    });
  });
});
