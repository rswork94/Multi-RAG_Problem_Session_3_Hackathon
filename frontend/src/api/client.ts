import type { SSEEvent } from "../types";

export async function getJSON<T = any>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await errText(r));
  return r.json();
}

export async function sendJSON<T = any>(url: string, method: "POST" | "PUT" | "DELETE", body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(await errText(r));
  return r.json();
}

async function errText(r: Response): Promise<string> {
  try {
    const j = await r.json();
    return typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
  } catch {
    return `${r.status} ${r.statusText}`;
  }
}

/** POST and read a Server-Sent Events stream ("data: {...}\n\n" lines). Resolves with the final event. */
export async function postSSE(
  url: string,
  body: unknown,
  onEvent: (e: SSEEvent) => void,
  signal?: AbortSignal,
): Promise<any> {
  const isForm = body instanceof FormData;
  const r = await fetch(url, {
    method: "POST",
    headers: isForm || body === undefined ? undefined : { "Content-Type": "application/json" },
    body: isForm ? (body as FormData) : body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) throw new Error(await errText(r));
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let final: any = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      const ev = JSON.parse(line.slice(6)) as SSEEvent;
      if (ev.type === "final") final = ev;
      if (ev.type === "error") {
        onEvent(ev);
        throw new Error(ev.message);
      }
      onEvent(ev);
    }
  }
  return final;
}
