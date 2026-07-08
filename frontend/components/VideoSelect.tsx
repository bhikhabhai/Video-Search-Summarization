import { useState } from "react";
import { VideoInfo } from "@/lib/types";

export default function VideoSelect({
  videos,
  value,
  onChange,
  onRefresh,
  onDelete,
}: {
  videos: VideoInfo[];
  value: string;
  onChange: (videoId: string) => void;
  onRefresh: () => void;
  onDelete?: (videoId: string) => Promise<void> | void;
}) {
  const [deleting, setDeleting] = useState(false);
  const current = videos.find((v) => v.video_id === value);

  async function handleDelete() {
    if (!value || !onDelete || deleting) return;
    const name = current?.filename ?? value;
    if (!window.confirm(`Remove "${name}" from the database? This cannot be undone.`)) {
      return;
    }
    setDeleting(true);
    try {
      await onDelete(value);
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="flex items-center gap-2">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="flex-1 rounded-lg border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm outline-none transition focus:border-emerald-500"
      >
        {videos.length === 0 && <option value="">No processed videos</option>}
        {videos.map((v) => (
          <option key={v.video_id} value={v.video_id}>
            {v.filename} · {v.duration_sec.toFixed(0)}s · {v.keyframe_count} frames
          </option>
        ))}
      </select>
      <button
        type="button"
        onClick={onRefresh}
        title="Refresh video list"
        aria-label="Refresh video list"
        className="flex h-10 w-10 items-center justify-center rounded-lg border border-neutral-700 bg-neutral-900 text-neutral-300 transition hover:border-neutral-500 hover:bg-neutral-800 hover:text-white"
      >
        <RefreshIcon />
      </button>
      {onDelete && (
        <button
          type="button"
          onClick={handleDelete}
          disabled={!value || deleting}
          title="Remove this video from the database"
          aria-label="Remove this video from the database"
          className="flex h-10 w-10 items-center justify-center rounded-lg border border-neutral-700 bg-neutral-900 text-neutral-400 transition hover:border-red-700 hover:bg-red-950/40 hover:text-red-400 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-neutral-700 disabled:hover:bg-neutral-900 disabled:hover:text-neutral-400"
        >
          {deleting ? <Spinner /> : <TrashIcon />}
        </button>
      )}
    </div>
  );
}

function RefreshIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
      <path d="M21 3v5h-5" />
      <path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
      <path d="M3 21v-5h5" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 6h18" />
      <path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
    </svg>
  );
}

function Spinner() {
  return (
    <svg
      className="animate-spin"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
    >
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
}
