"use client";

import { useRef, useState } from "react";
import { getFrames, getStatus, ingestVideo } from "@/lib/api";
import { Frame } from "@/lib/types";
import Gallery from "./Gallery";

export default function UploadTab({
  onIngested,
}: {
  onIngested: (videoId: string) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [frames, setFrames] = useState<Frame[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  function pick(f: File | null | undefined) {
    if (!f) return;
    setFile(f);
    setError("");
    setFrames([]);
    setProgress(0);
    setMessage("");
  }

  async function handleUpload() {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    setFrames([]);
    setProgress(0);
    setMessage("Uploading…");

    try {
      const { job_id, video_id } = await ingestVideo(file);
      setMessage("Queued…");

      // Poll status once per second until done or error.
      // eslint-disable-next-line no-constant-condition
      while (true) {
        await new Promise((r) => setTimeout(r, 1000));
        const s = await getStatus(job_id);
        setProgress(s.progress);
        setMessage(s.message);
        if (s.state === "error") {
          setError(s.error || "Ingestion failed");
          break;
        }
        if (s.state === "done") {
          setFrames(await getFrames(video_id));
          onIngested(video_id);
          break;
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pick(e.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-12 text-center transition ${
          dragging
            ? "border-emerald-500 bg-emerald-500/5"
            : "border-neutral-700 hover:border-neutral-500 hover:bg-neutral-900/50"
        }`}
      >
        <div className="text-3xl">🎬</div>
        <p className="text-sm text-neutral-300">
          {file ? (
            <span className="font-medium text-emerald-400">{file.name}</span>
          ) : (
            <>
              <span className="font-medium">Drop a video here</span> or click to
              browse
            </>
          )}
        </p>
        <p className="text-xs text-neutral-500">≤ 8 min 20 s · ≤ 300 MB</p>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => pick(e.target.files?.[0])}
        />
      </div>

      <button
        onClick={handleUpload}
        disabled={!file || busy}
        className="w-full rounded-lg bg-emerald-600 px-4 py-2.5 font-medium text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {busy ? "Processing…" : "Ingest video"}
      </button>

      {(busy || progress > 0) && !error && (
        <div className="space-y-1.5">
          <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-800">
            <div
              className="h-full rounded-full bg-emerald-500 transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-neutral-400">
            <span className="font-mono text-emerald-400">{progress}%</span>{" "}
            {message}
          </p>
        </div>
      )}

      {error && (
        <p className="rounded-lg border border-red-900 bg-red-950/50 px-3 py-2 text-sm text-red-300">
          ❌ {error}
        </p>
      )}

      {frames.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-neutral-300">
            Extracted keyframes ({frames.length})
          </h3>
          <Gallery frames={frames} columns={5} maxHeight="24rem" />
        </div>
      )}
    </div>
  );
}
