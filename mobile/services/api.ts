/**
 * API client for the Eye Hear U backend.
 *
 * ## Configure the backend URL
 *
 * URL resolution order (first non-empty value wins):
 *  1. `extra.apiBaseUrl` in `app.json` — runtime config via expo-constants.
 *     Set this to override the URL without rebuilding (e.g. in `eas.json` profiles).
 *     This is the only override that works in both dev and non-dev (staging/release) builds.
 *  2. (dev builds only — `__DEV__ === true`) `EXPO_PUBLIC_API_URL` env var, inlined at
 *     bundle time; set in `mobile/.env`.
 *     - Same LAN: `http://192.168.x.x:8000` (uvicorn with `--host 0.0.0.0`)
 *     - Tunnel: run `npx localtunnel --port 8000`, paste the `https://….loca.lt` URL.
 *  3. (dev builds only) `http://127.0.0.1:8000` — simulator default.
 *  4. (non-dev builds) `https://api.eyehearu.app` — production default.
 *
 * LocalTunnel can be unreliable; same Wi‑Fi + LAN IP is usually more stable for demos.
 * iOS: if Expo Go cannot load the bundle, enable Local Network access for Expo Go under
 * Settings → Privacy & Security → Local Network.
 */

import Constants from "expo-constants";

/** Strip trailing slash */
function normalizeBaseUrl(url: string): string {
  return url.replace(/\/+$/, "");
}

function resolveApiBaseUrl(): string {
  // 1. Runtime config via expo-constants (app.json → extra.apiBaseUrl)
  const fromConstants = (
    Constants.expoConfig?.extra?.apiBaseUrl as string | undefined
  )?.trim();
  if (fromConstants) {
    return normalizeBaseUrl(fromConstants);
  }

  if (!__DEV__) {
    return "https://api.eyehearu.app";
  }
  const fromEnv = process.env.EXPO_PUBLIC_API_URL?.trim();
  if (fromEnv) {
    return normalizeBaseUrl(fromEnv);
  }
  // Simulator can reach host machine; physical device needs .env with LAN IP or tunnel URL.
  return "http://127.0.0.1:8000";
}

export const API_BASE_URL = resolveApiBaseUrl();

const EXTRA_HEADERS: Record<string, string> = API_BASE_URL.includes("loca.lt")
  ? { "bypass-tunnel-reminder": "true" }
  : {};

export interface TopKPrediction {
  sign: string;
  confidence: number;
}

export interface PredictionResult {
  sign: string;
  confidence: number;
  top_k: TopKPrediction[];
  message?: string;
}

/** One clip’s classifier output in a multi-clip sentence request. */
export interface SentenceClipResult {
  top_k: TopKPrediction[];
}

/** One beam hypothesis for the full gloss sequence. */
export interface SentenceBeamRow {
  glosses: string[];
  score: number;
  english: string;
}

/** Response from POST /api/v1/predict/sentence (beam + gloss LM). */
export interface SentencePredictionResult {
  clips: SentenceClipResult[];
  beam: SentenceBeamRow[];
  best_glosses: string[];
  english: string;
}

/** True when response looks like a dead / overloaded LocalTunnel. */
export function isTunnelUnavailable(status: number, body: string): boolean {
  if (status !== 503 && status !== 502) return false;
  const b = body.toLowerCase();
  return (
    b.includes("tunnel") ||
    b.includes("unavailable") ||
    b.includes("loca.lt") ||
    b.includes("localtunnel")
  );
}

/**
 * Human-readable hint when tunnel or network fails.
 */
export function explainApiFailure(status: number, body: string): string {
  if (isTunnelUnavailable(status, body)) {
    return (
      "Tunnel expired or unavailable. On the machine running the API, run:\n" +
      "  npx localtunnel --port 8000\n" +
      "Then set the new https URL in mobile/.env as EXPO_PUBLIC_API_URL=… and restart Expo.\n" +
      "Or use the API host’s LAN IP on the same Wi‑Fi as the device instead of a tunnel."
    );
  }
  if (status === 503) {
    return "Server unavailable (503). Is the backend running? Try: uvicorn app.main:app --host 0.0.0.0 --port 8000";
  }
  return body.slice(0, 200) || `HTTP ${status}`;
}

/**
 * Send a recorded video clip to the backend for ASL sign prediction.
 */
export async function predictSign(videoUri: string): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", {
    uri: videoUri,
    name: "clip.mp4",
    type: "video/mp4",
  } as any);

  const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: "POST",
    body: formData,
    headers: { ...EXTRA_HEADERS },
  });

  const text = await response.text();

  if (!response.ok) {
    const hint = explainApiFailure(response.status, text);
    throw new Error(`Prediction failed (${response.status}): ${hint}`);
  }

  return JSON.parse(text) as PredictionResult;
}

/**
 * Send multiple recorded clips in order; backend runs batched I3D + beam search + gloss LM.
 */
export async function predictSentence(
  videoUris: string[],
  opts?: { beamSize?: number; lmWeight?: number; topK?: number }
): Promise<SentencePredictionResult> {
  if (videoUris.length === 0) {
    throw new Error("Add at least one video clip.");
  }
  const formData = new FormData();
  videoUris.forEach((uri, i) => {
    formData.append("files", {
      uri,
      name: `clip_${i}.mp4`,
      type: "video/mp4",
    } as any);
  });

  const params = new URLSearchParams();
  if (opts?.beamSize != null) {
    params.set("beam_size", String(opts.beamSize));
  }
  if (opts?.lmWeight != null) {
    params.set("lm_weight", String(opts.lmWeight));
  }
  if (opts?.topK != null) {
    params.set("top_k", String(opts.topK));
  }
  const q = params.toString();
  const url = `${API_BASE_URL}/api/v1/predict/sentence${q ? `?${q}` : ""}`;

  const response = await fetch(url, {
    method: "POST",
    body: formData,
    headers: { ...EXTRA_HEADERS },
  });

  const text = await response.text();

  if (!response.ok) {
    const hint = explainApiFailure(response.status, text);
    throw new Error(`Sentence prediction failed (${response.status}): ${hint}`);
  }

  return JSON.parse(text) as SentencePredictionResult;
}

export interface HealthResult {
  alive: boolean;
  modelLoaded: boolean;
  /** LocalTunnel (or similar) returned 502/503 — URL likely expired. */
  tunnelUnavailable: boolean;
}

/**
 * Check if the backend is reachable and the model is loaded.
 */
export async function checkHealth(): Promise<HealthResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/ready`, {
      headers: { ...EXTRA_HEADERS },
    });
    const text = await response.text();

    if (!response.ok) {
      return {
        alive: false,
        modelLoaded: false,
        tunnelUnavailable: isTunnelUnavailable(response.status, text),
      };
    }

    let data: { model_loaded?: boolean };
    try {
      data = JSON.parse(text);
    } catch {
      return { alive: false, modelLoaded: false, tunnelUnavailable: false };
    }

    return {
      alive: true,
      modelLoaded: data.model_loaded === true,
      tunnelUnavailable: false,
    };
  } catch {
    return { alive: false, modelLoaded: false, tunnelUnavailable: false };
  }
}
