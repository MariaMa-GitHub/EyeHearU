import React from "react";
import { render } from "@testing-library/react-native";

jest.mock("expo-status-bar", () => ({
  StatusBar: () => null,
}));

jest.mock("expo-router", () => {
  const React = require("react");
  const { View } = require("react-native");
  const Stack = ({ children }: { children: React.ReactNode }) => (
    <View testID="stack-root">{children}</View>
  );
  const Screen = () => <View testID="stack-screen" />;
  return { Stack: Object.assign(Stack, { Screen }) };
});

import RootLayout from "../app/_layout";

describe("RootLayout", () => {
  it("renders stack navigator and status bar slot", () => {
    const { getByTestId } = render(<RootLayout />);
    expect(getByTestId("stack-root")).toBeTruthy();
  });
});
