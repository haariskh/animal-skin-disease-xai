import type { AnalyzeResponse, Health, Metadata } from "./types";

/**
 * Requests go to a relative path so the same build works two ways: behind the Vite dev
 * proxy on :5173, and served directly by uvicorn once the frontend has been built.
 */
const BASE = "/api";

/** The server replies with {detail: "..."} on error; surface that rather than a status code. */
async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) {
    return (await response.json()) as T;
  }
  let detail = `Request failed with status ${response.status}`;
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") {
      detail = body.detail;
    }
  } catch {
    // Response had no JSON body; keep the status-based message.
  }
  throw new Error(detail);
}

export async function fetchHealth(): Promise<Health> {
  return unwrap<Health>(await fetch(`${BASE}/health`));
}

export async function fetchMetadata(): Promise<Metadata> {
  return unwrap<Metadata>(await fetch(`${BASE}/metadata`));
}

export async function analyzeImage(options: {
  file: File;
  methods?: string[];
  targetClass?: string;
  includeFaithfulness?: boolean;
  signal?: AbortSignal;
}): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", options.file);
  if (options.methods?.length) {
    form.append("methods", options.methods.join(","));
  }
  if (options.targetClass) {
    form.append("target_class", options.targetClass);
  }
  form.append("include_faithfulness", String(options.includeFaithfulness ?? true));

  const response = await fetch(`${BASE}/analyze`, {
    method: "POST",
    body: form,
    signal: options.signal,
  });
  return unwrap<AnalyzeResponse>(response);
}
