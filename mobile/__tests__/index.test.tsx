import React from "react";
import { render, screen, fireEvent } from "@testing-library/react-native";
import { router } from "expo-router";

jest.mock("expo-router", () => ({
  router: { push: jest.fn() },
}));

import HomeScreen from "../app/index";

describe("HomeScreen", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders hero and navigates to camera and history", () => {
    render(<HomeScreen />);

    expect(screen.getByText("ASL to English, one sign at a time")).toBeTruthy();

    fireEvent.press(screen.getByText("Start Translating"));
    expect(router.push).toHaveBeenCalledWith("/camera");

    fireEvent.press(screen.getByText("View History"));
    expect(router.push).toHaveBeenCalledWith("/history");
  });
});
