// Mirrors the Pydantic models exposed by the FastAPI backend (src/models.py).

export interface VideoInfo {
  video_id: string;
  filename: string;
  duration_sec: number;
  keyframe_count: number;
}

export interface Frame {
  video_id: string;
  frame_idx: number;
  timestamp_sec: number;
  caption: string;
  detection_labels: string[];
  thumbnail_b64: string;
  score: number;
}

export type JobState = "queued" | "processing" | "done" | "error";

export interface IngestStatus {
  job_id: string;
  video_id: string;
  state: JobState;
  progress: number;
  message: string;
  error?: string | null;
  keyframe_count: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}
