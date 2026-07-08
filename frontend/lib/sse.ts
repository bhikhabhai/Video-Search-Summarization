// Minimal Server-Sent-Events reader over a fetch() streaming Response.
//
// The backend emits frames of the form:
//     data: <payload>\n\n
// where <payload> is either the literal string "[DONE]" or a JSON-encoded
// value (see src/api/main.py::_sse). This reads the byte stream, splits it
// into SSE records, and hands each raw `data:` payload to the callback.

export async function readSSE(
  res: Response,
  onData: (payload: string) => void,
): Promise<void> {
  if (!res.body) return;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const record = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of record.split("\n")) {
        if (line.startsWith("data: ")) {
          onData(line.slice(6));
        }
      }
    }
  }
}
