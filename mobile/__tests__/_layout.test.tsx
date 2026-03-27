/**
 * Tests for root layout (app/_layout.tsx).
 */

import React from "react";
import { render, screen } from "@testing-library/react-native";

jest.mock("expo-status-bar", () => ({
  StatusBar: () => null,
}));

jest.mock("expo-router", () => {
  const React = require("react");
  const { View } = require("react-native");
  function Stack({ children }: { children?: React.ReactNode }) {
    return <View testID="root-stack">{children}</View>;
  }
  Stack.Screen = () => null;
  return { Stack };
});

import RootLayout from "../app/_layout";

describe("RootLayout", () => {
  it("renders the navigation stack", () => {
    render(<RootLayout />);
    expect(screen.getByTestId("root-stack")).toBeTruthy();
  });
});
