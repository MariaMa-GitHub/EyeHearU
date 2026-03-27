/**
 * Tests for the API service (services/api.ts).
 *
 * We re-export the pure helpers directly from the module and mock `fetch`
 * globally so that `predictSign` and `checkHealth` can be tested without
 * hitting a real server.
 */

jest.mock("expo-constants", () => ({
  expoConfig: { extra: {} },
}));

import {
  isTunnelUnavailable,
  explainApiFailure,
  predictSign,
  checkHealth,
} from "../services/api";

/* ------------------------------------------------------------------ */
/*  Helpers: normalizeBaseUrl is not exported, but we can test it      */
/*  indirectly via API_BASE_URL.  The three pure exported helpers are  */
/*  tested directly below.                                             */
/* ------------------------------------------------------------------ */

// We need to test normalizeBaseUrl. Since it is not exported, we
// re-require the module internals via a small trick: we extract it
// by reading the source. Instead, let's test indirectly and also
// test the exported helpers thoroughly.

/* ------------------------------------------------------------------ */
/*  isTunnelUnavailable                                                */
/* ------------------------------------------------------------------ */
describe("isTunnelUnavailable", () => {
  it("returns false for non-502/503 status codes", () => {
    expect(isTunnelUnavailable(200, "tunnel")).toBe(false);
    expect(isTunnelUnavailable(404, "tunnel unavailable")).toBe(false);
    expect(isTunnelUnavailable(500, "loca.lt")).toBe(false);
  });

  it("returns true for 503 with tunnel-related body", () => {
    expect(isTunnelUnavailable(503, "tunnel is down")).toBe(true);
    expect(isTunnelUnavailable(503, "Service Unavailable")).toBe(true);
    expect(isTunnelUnavailable(503, "loca.lt error")).toBe(true);
    expect(isTunnelUnavailable(503, "localtunnel proxy")).toBe(true);
  });

  it("returns true for 502 with tunnel-related body", () => {
    expect(isTunnelUnavailable(502, "Tunnel expired")).toBe(true);
    expect(isTunnelUnavailable(502, "UNAVAILABLE")).toBe(true);
  });

  it("returns false for 503/502 with non-tunnel body", () => {
    expect(isTunnelUnavailable(503, "internal server error")).toBe(false);
    expect(isTunnelUnavailable(502, "bad gateway")).toBe(false);
  });

  it("performs case-insensitive matching on body", () => {
    expect(isTunnelUnavailable(503, "TUNNEL")).toBe(true);
    expect(isTunnelUnavailable(502, "LOCA.LT")).toBe(true);
    expect(isTunnelUnavailable(503, "LocalTunnel")).toBe(true);
  });
});

/* ------------------------------------------------------------------ */
/*  explainApiFailure                                                  */
/* ------------------------------------------------------------------ */
describe("explainApiFailure", () => {
  it("returns tunnel hint when tunnel is unavailable", () => {
    const msg = explainApiFailure(503, "tunnel expired");
    expect(msg).toContain("Tunnel expired or unavailable");
    expect(msg).toContain("npx localtunnel --port 8000");
    expect(msg).toContain("EXPO_PUBLIC_API_URL");
  });

  it("returns generic 503 message when not tunnel-related", () => {
    const msg = explainApiFailure(503, "internal error");
    expect(msg).toContain("Server unavailable (503)");
    expect(msg).toContain("uvicorn");
  });

  it("returns truncated body for other status codes", () => {
    const msg = explainApiFailure(400, "Bad request body");
    expect(msg).toBe("Bad request body");
  });

  it("returns HTTP status when body is empty", () => {
    expect(explainApiFailure(500, "")).toBe("HTTP 500");
  });

  it("truncates long bodies to 200 characters", () => {
    const longBody = "x".repeat(300);
    const msg = explainApiFailure(400, longBody);
    expect(msg.length).toBe(200);
  });
});

/* ------------------------------------------------------------------ */
/*  predictSign                                                        */
/* ------------------------------------------------------------------ */
describe("predictSign", () => {
  const mockFetch = jest.fn();

  beforeEach(() => {
    jest.resetAllMocks();
    global.fetch = mockFetch;
  });

  it("sends a POST with FormData and returns parsed result", async () => {
    const mockResult = {
      sign: "hello",
      confidence: 0.95,
      top_k: [{ sign: "hello", confidence: 0.95 }],
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: async () => JSON.stringify(mockResult),
    });

    const result = await predictSign("file:///video.mp4");

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, options] = mockFetch.mock.calls[0];
    expect(url).toContain("/api/v1/predict");
    expect(options.method).toBe("POST");
    expect(options.body).toBeInstanceOf(FormData);
    expect(result).toEqual(mockResult);
  });

  it("throws with explanatory message on non-ok response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      text: async () => "tunnel expired",
    });

    await expect(predictSign("file:///video.mp4")).rejects.toThrow(
      /Prediction failed \(503\)/
    );
  });

  it("throws on network error", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Network error"));

    await expect(predictSign("file:///video.mp4")).rejects.toThrow(
      "Network error"
    );
  });
});

/* ------------------------------------------------------------------ */
/*  checkHealth                                                        */
/* ------------------------------------------------------------------ */
describe("checkHealth", () => {
  const mockFetch = jest.fn();

  beforeEach(() => {
    jest.resetAllMocks();
    global.fetch = mockFetch;
  });

  it("returns alive + modelLoaded when backend responds with model_loaded: true", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({ model_loaded: true }),
    });

    const result = await checkHealth();
    expect(result).toEqual({
      alive: true,
      modelLoaded: true,
      tunnelUnavailable: false,
    });
  });

  it("returns modelLoaded false when model_loaded is false", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({ model_loaded: false }),
    });

    const result = await checkHealth();
    expect(result).toEqual({
      alive: true,
      modelLoaded: false,
      tunnelUnavailable: false,
    });
  });

  it("returns not alive when response is not ok", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "server error",
    });

    const result = await checkHealth();
    expect(result).toEqual({
      alive: false,
      modelLoaded: false,
      tunnelUnavailable: false,
    });
  });

  it("detects tunnel unavailable on 503 with tunnel body", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      text: async () => "tunnel unavailable",
    });

    const result = await checkHealth();
    expect(result).toEqual({
      alive: false,
      modelLoaded: false,
      tunnelUnavailable: true,
    });
  });

  it("returns not alive when fetch throws (network error)", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Network failure"));

    const result = await checkHealth();
    expect(result).toEqual({
      alive: false,
      modelLoaded: false,
      tunnelUnavailable: false,
    });
  });

  it("returns not alive when response body is not valid JSON", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: async () => "not json",
    });

    const result = await checkHealth();
    expect(result).toEqual({
      alive: false,
      modelLoaded: false,
      tunnelUnavailable: false,
    });
  });

  it("calls the /ready endpoint", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({ model_loaded: true }),
    });

    await checkHealth();
    expect(mockFetch.mock.calls[0][0]).toContain("/ready");
  });
});

/* ------------------------------------------------------------------ */
/*  resolveApiBaseUrl branches                                         */
/* ------------------------------------------------------------------ */
describe("resolveApiBaseUrl", () => {
  it("uses Constants.expoConfig.extra.apiBaseUrl when set (with trailing slash stripped)", () => {
    // Re-require the module with a fresh Constants mock that has apiBaseUrl
    jest.resetModules();
    jest.mock("expo-constants", () => ({
      expoConfig: { extra: { apiBaseUrl: "https://custom.api.example.com/" } },
    }));
    const { API_BASE_URL } = require("../services/api");
    expect(API_BASE_URL).toBe("https://custom.api.example.com");
  });

  it("returns production URL when __DEV__ is false and no Constants override", () => {
    jest.resetModules();
    jest.mock("expo-constants", () => ({
      expoConfig: { extra: {} },
    }));
    const originalDev = (global as any).__DEV__;
    (global as any).__DEV__ = false;
    const { API_BASE_URL } = require("../services/api");
    expect(API_BASE_URL).toBe("https://api.eyehearu.app");
    (global as any).__DEV__ = originalDev;
  });

  it("uses EXPO_PUBLIC_API_URL env var when __DEV__ is true and no Constants override", () => {
    jest.resetModules();
    jest.mock("expo-constants", () => ({
      expoConfig: { extra: {} },
    }));
    const originalDev = (global as any).__DEV__;
    (global as any).__DEV__ = true;
    const originalEnv = process.env.EXPO_PUBLIC_API_URL;
    process.env.EXPO_PUBLIC_API_URL = "http://192.168.1.100:8000/";
    const { API_BASE_URL } = require("../services/api");
    expect(API_BASE_URL).toBe("http://192.168.1.100:8000");
    process.env.EXPO_PUBLIC_API_URL = originalEnv;
    (global as any).__DEV__ = originalDev;
  });

  it("normalizeBaseUrl strips multiple trailing slashes", () => {
    jest.resetModules();
    jest.mock("expo-constants", () => ({
      expoConfig: { extra: { apiBaseUrl: "https://example.com///" } },
    }));
    const { API_BASE_URL } = require("../services/api");
    expect(API_BASE_URL).toBe("https://example.com");
  });

  it("sends bypass-tunnel-reminder header when apiBaseUrl uses loca.lt", async () => {
    jest.resetModules();
    jest.doMock("expo-constants", () => ({
      expoConfig: { extra: { apiBaseUrl: "https://abc.loca.lt/" } },
    }));
    const mockFetch = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () =>
        JSON.stringify({
          sign: "hello",
          confidence: 0.9,
          top_k: [{ sign: "hello", confidence: 0.9 }],
        }),
    });
    global.fetch = mockFetch as unknown as typeof fetch;

    const { predictSign } = require("../services/api");
    await predictSign("file:///clip.mp4");

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [, options] = mockFetch.mock.calls[0];
    expect(options.headers).toEqual({ "bypass-tunnel-reminder": "true" });
    expect(mockFetch.mock.calls[0][0]).toContain("abc.loca.lt");
  });
});
