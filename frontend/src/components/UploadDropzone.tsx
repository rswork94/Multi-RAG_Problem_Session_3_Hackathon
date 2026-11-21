import { useRef, useState } from "react";

const OK = [".pdf", ".docx", ".json", ".csv", ".txt", ".md"];
const MAX_MB = 25;

/** Drag-and-drop upload with client-side type/size checks. */
export function UploadDropzone({ onFiles, disabled }: { onFiles: (files: File[]) => void; disabled?: boolean }) {
  const [over, setOver] = useState(false);
  const [err, setErr] = useState("");
  const input = useRef<HTMLInputElement>(null);

  const take = (list: FileList | null) => {
    if (!list) return;
    const files = Array.from(list);
    const bad = files.filter((f) => !OK.some((e) => f.name.toLowerCase().endsWith(e)) || f.size > MAX_MB * 1024 * 1024);
    if (bad.length) {
      setErr(`Skipped ${bad.map((f) => f.name).join(", ")}: PDF/DOCX/JSON/CSV/TXT/MD up to ${MAX_MB} MB only.`);
    } else {
      setErr("");
    }
    const good = files.filter((f) => !bad.includes(f));
    if (good.length) onFiles(good);
  };

  return (
    <div>
      <div
        className={`drop ${over ? "over" : ""}`}
        onClick={() => !disabled && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (!disabled) take(e.dataTransfer.files);
        }}
      >
        Drop PDF / DOCX / JSON / CSV here, or click to choose
        <input ref={input} type="file" multiple hidden accept={OK.join(",")} onChange={(e) => take(e.target.files)} />
      </div>
      {err && <div className="small" style={{ color: "var(--warn)", marginTop: 4 }}>{err}</div>}
    </div>
  );
}
