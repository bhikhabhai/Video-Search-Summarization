import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Mini-VSS — Video Search & Summarization",
  description:
    "Upload a video, then ask questions, retrieve semantic clips, and stream summaries.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
