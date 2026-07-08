"use client";

import { useCallback, useEffect, useState } from "react";
import AskTab from "@/components/AskTab";
import SummaryTab from "@/components/SummaryTab";
import UploadTab from "@/components/UploadTab";
import { deleteVideo, listVideos } from "@/lib/api";
import { VideoInfo } from "@/lib/types";

type Tab = "upload" | "ask" | "summary";

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: "upload", label: "Upload", icon: "📤" },
  { id: "ask", label: "Ask", icon: "💬" },
  { id: "summary", label: "Summary", icon: "📝" },
];

export default function Home() {
  const [tab, setTab] = useState<Tab>("upload");
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [selected, setSelected] = useState<string>("");

  const refresh = useCallback(async () => {
    const list = await listVideos();
    setVideos(list);
    setSelected((cur) => cur || list[0]?.video_id || "");
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function handleIngested(videoId: string) {
    refresh();
    setSelected(videoId);
  }

  async function handleDelete(videoId: string) {
    await deleteVideo(videoId);
    const list = await listVideos();
    setVideos(list);
    // If we just removed the selected video, fall back to the first remaining.
    setSelected((cur) => (cur === videoId ? list[0]?.video_id || "" : cur));
  }

  return (
    <main className="mx-auto max-w-3xl px-4 py-10">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold tracking-tight">
          🎬 Mini-VSS
        </h1>
        <p className="mt-1 text-sm text-neutral-400">
          Video Search &amp; Summarization
        </p>
      </header>

      <nav className="mb-6 flex gap-1 rounded-xl border border-neutral-800 bg-neutral-900/60 p-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium transition ${
              tab === t.id
                ? "bg-neutral-800 text-white shadow"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <span className="mr-1.5">{t.icon}</span>
            {t.label}
          </button>
        ))}
      </nav>

      <section className="rounded-2xl border border-neutral-800 bg-neutral-900/30 p-6">
        {tab === "upload" && <UploadTab onIngested={handleIngested} />}
        {tab === "ask" && (
          <AskTab
            videos={videos}
            selected={selected}
            onSelect={setSelected}
            onRefresh={refresh}
            onDelete={handleDelete}
          />
        )}
        {tab === "summary" && (
          <SummaryTab
            videos={videos}
            selected={selected}
            onSelect={setSelected}
            onRefresh={refresh}
            onDelete={handleDelete}
          />
        )}
      </section>

      <footer className="mt-8 text-center text-xs text-neutral-600">
        Built with Next.js · FastAPI backend
      </footer>
    </main>
  );
}
