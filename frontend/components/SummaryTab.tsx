"use client";

import { useState } from "react";
import { getFrames, streamSummary } from "@/lib/api";
import { Frame, VideoInfo } from "@/lib/types";
import Gallery from "./Gallery";
import VideoSelect from "./VideoSelect";

export default function SummaryTab({
  videos,
  selected,
  onSelect,
  onRefresh,
  onDelete,
}: {
  videos: VideoInfo[];
  selected: string;
  onSelect: (id: string) => void;
  onRefresh: () => void;
  onDelete: (id: string) => Promise<void> | void;
}) {
  const [summary, setSummary] = useState("");
  const [frames, setFrames] = useState<Frame[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function generate() {
    if (!selected || busy) return;
    setBusy(true);
    setError("");
    setSummary("");
    setFrames([]);

    try {
      setFrames(await getFrames(selected));
      await streamSummary(selected, (token) =>
        setSummary((s) => s + token),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <VideoSelect
        videos={videos}
        value={selected}
        onChange={onSelect}
        onRefresh={onRefresh}
        onDelete={onDelete}
      />

      <button
        onClick={generate}
        disabled={busy || !selected}
        className="w-full rounded-lg bg-emerald-600 px-4 py-2.5 font-medium text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {busy ? "Summarising…" : "Generate summary"}
      </button>

      {error && (
        <p className="rounded-lg border border-red-900 bg-red-950/50 px-3 py-2 text-sm text-red-300">
          ❌ {error}
        </p>
      )}

      {(summary || busy) && (
        <div className="min-h-[6rem] whitespace-pre-wrap rounded-xl border border-neutral-800 bg-neutral-900/40 p-4 text-sm leading-relaxed text-neutral-200">
          {summary || "⏳ summarising…"}
        </div>
      )}

      {frames.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-neutral-300">Timeline</h3>
          <Gallery frames={frames} columns={6} maxHeight="18rem" />
        </div>
      )}
    </div>
  );
}
