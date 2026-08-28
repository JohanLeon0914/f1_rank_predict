import type {
  ApiHealth,
  LocalF1Data,
  PredictionRequest,
  PredictionResponse,
} from "./types";

async function readJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = payload?.detail ?? payload?.error ?? "Error inesperado";
    const message =
      typeof detail === "string" ? detail : JSON.stringify(detail, null, 2);
    throw new Error(message);
  }

  return payload as T;
}

export async function getHealth(): Promise<ApiHealth> {
  return readJson<ApiHealth>(await fetch("/api/ml/health", { cache: "no-store" }));
}

export async function getMetrics(): Promise<Record<string, unknown>> {
  return readJson<Record<string, unknown>>(
    await fetch("/api/ml/metrics", { cache: "no-store" }),
  );
}

export async function predictRace(
  payload: PredictionRequest,
): Promise<PredictionResponse> {
  return readJson<PredictionResponse>(
    await fetch("/api/ml/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

export async function getLocalF1Data(): Promise<LocalF1Data> {
  return readJson<LocalF1Data>(
    await fetch("/api/f1/local", { cache: "no-store" }),
  );
}
