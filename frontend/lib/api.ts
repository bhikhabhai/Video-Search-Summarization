import { readSSE } from "./sse";
import { Frame, IngestStatus, VideoInfo } from "./types";

function resolveApiUrl(): string {
  // Use Next.js same-origin proxy to avoid LAN browser -> API direct calls
  // (CORS/mixed-content/firewall issues) and keep one public origin (:3000).
  return "/api";
}

export const API_URL = resolveApiUrl();

// Marker the /query endpoint prepends to the supporting-frames payload.
const FRAMES_MARKER = "\u0000FRAMES:";

export function thumbSrc(b64: string): string {
  return `data:image/jpeg;base64,${b64}`;
}

export async function listVideos(): Promise<VideoInfo[]> {
  try {
    const res = await fetch(`${API_URL}/videos`, { cache: "no-store" });
    if (!res.ok) return [];
    return await res.json();
  } catch {
    return [];
  }
}

export async function ingestVideo(
  file: File,
  signal?: AbortSignal,
): Promise<{ job_id: string; video_id: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/ingest`, {
    method: "POST",
    body: form,
    signal,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getStatus(
  jobId: string,
  signal?: AbortSignal,
): Promise<IngestStatus> {
  const res = await fetch(`${API_URL}/status/${jobId}`, {
    cache: "no-store",
    signal,
  });
  if (!res.ok) throw new Error(`status request failed (${res.status})`);
  return res.json();
}

export async function getFrames(videoId: string): Promise<Frame[]> {
  const res = await fetch(`${API_URL}/frames/${videoId}`, { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

export async function deleteVideo(videoId: string): Promise<void> {
  const res = await fetch(`${API_URL}/videos/${videoId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
}

export async function streamQuery(
  videoId: string,
  question: string,
  onToken: (token: string) => void,
  onFrames: (frames: Frame[]) => void,
): Promise<void> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId, question }),
  });
  if (!res.ok) throw new Error(await res.text());

  await readSSE(res, (payload) => {
    if (payload === "[DONE]") return;
    let value: unknown;
    try {
      value = JSON.parse(payload);
    } catch {
      return;
    }
    if (typeof value !== "string") return;
    if (value.startsWith(FRAMES_MARKER)) {
      try {
        onFrames(JSON.parse(value.slice(FRAMES_MARKER.length)) as Frame[]);
      } catch {
        /* ignore malformed frame payload */
      }
      return;
    }
    onToken(value);
  });
}

export async function streamSummary(
  videoId: string,
  onToken: (token: string) => void,
): Promise<void> {
  const res = await fetch(`${API_URL}/summary/${videoId}`);
  if (!res.ok) throw new Error(await res.text());

  await readSSE(res, (payload) => {
    if (payload === "[DONE]") return;
    try {
      const token = JSON.parse(payload);
      if (typeof token === "string") onToken(token);
    } catch {
      /* ignore */
    }
  });
}
