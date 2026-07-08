"use client";

import { useRef, useState } from "react";
import { streamQuery } from "@/lib/api";
import { ChatMessage, Frame, VideoInfo } from "@/lib/types";
import Gallery from "./Gallery";
import VideoSelect from "./VideoSelect";

export default function AskTab({
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
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  function scrollToBottom() {
    requestAnimationFrame(() => {
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }

  async function ask() {
    const q = question.trim();
    if (!q || busy) return;
    if (!selected) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: "Pick a processed video first." },
      ]);
      return;
    }

    setBusy(true);
    setQuestion("");
    setFrames([]);
    setMessages((m) => [
      ...m,
      { role: "user", content: q },
      { role: "assistant", content: "" },
    ]);
    scrollToBottom();

    try {
      await streamQuery(
        selected,
        q,
        (token) => {
          setMessages((m) => {
            const next = [...m];
            next[next.length - 1] = {
              role: "assistant",
              content: next[next.length - 1].content + token,
            };
            return next;
          });
          scrollToBottom();
        },
        (f) => setFrames(f),
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setMessages((m) => {
        const next = [...m];
        next[next.length - 1] = {
          role: "assistant",
          content: `[error contacting API: ${msg}]`,
        };
        return next;
      });
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

      <div
        ref={scrollRef}
        className="custom-scroll h-80 space-y-3 overflow-y-auto rounded-xl border border-neutral-800 bg-neutral-900/40 p-4"
      >
        {messages.length === 0 ? (
          <p className="pt-24 text-center text-sm text-neutral-500">
            Ask something like{" "}
            <em>“when does a person appear in a red jacket?”</em>
          </p>
        ) : (
          messages.map((m, i) => (
            <div
              key={i}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm ${
                  m.role === "user"
                    ? "bg-emerald-600 text-white"
                    : "bg-neutral-800 text-neutral-100"
                }`}
              >
                {m.content || (busy ? "…" : "")}
              </div>
            </div>
          ))
        )}
      </div>

      <div className="flex gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              ask();
            }
          }}
          placeholder="Ask a question about the video…"
          disabled={busy}
          className="flex-1 rounded-lg border border-neutral-700 bg-neutral-900 px-3 py-2.5 text-sm outline-none transition focus:border-emerald-500 disabled:opacity-50"
        />
        <button
          onClick={ask}
          disabled={busy || !question.trim()}
          className="rounded-lg bg-emerald-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "…" : "Ask"}
        </button>
      </div>

      {frames.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-neutral-300">
            Retrieved frames
          </h3>
          <Gallery frames={frames} columns={3} maxHeight="16rem" />
        </div>
      )}
    </div>
  );
}
