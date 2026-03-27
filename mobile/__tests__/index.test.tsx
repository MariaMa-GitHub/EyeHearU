/**
 * Tests for home screen (app/index.tsx).
 */

import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react-native";
import { router } from "expo-router";

jest.mock("expo-router", () => ({
  router: { push: jest.fn() },
}));

import HomeScreen from "../app/index";

const mockPush = router.push as jest.Mock;

describe("HomeScreen", () => {
  beforeEach(() => {
    mockPush.mockClear();
  });

  it("renders title and navigation actions", () => {
    render(<HomeScreen />);
    expect(screen.getByText(/Eye/)).toBeTruthy();
    expect(screen.getByText("ASL to English, one sign at a time")).toBeTruthy();
    expect(screen.getByText("Start Translating")).toBeTruthy();
    expect(screen.getByText("View History")).toBeTruthy();
  });

  it("navigates to camera when Start Translating is pressed", () => {
    render(<HomeScreen />);
    fireEvent.press(screen.getByText("Start Translating"));
    expect(mockPush).toHaveBeenCalledWith("/camera");
  });

  it("navigates to history when View History is pressed", () => {
    render(<HomeScreen />);
    fireEvent.press(screen.getByText("View History"));
    expect(mockPush).toHaveBeenCalledWith("/history");
  });

  it("runs hand animation loop on mount", () => {
    jest.useFakeTimers();
    render(<HomeScreen />);
    act(() => {
      jest.advanceTimersByTime(3500);
    });
    jest.useRealTimers();
  });
});
