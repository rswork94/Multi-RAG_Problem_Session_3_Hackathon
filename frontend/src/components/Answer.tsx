import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Markdown answer where [n] / [1, 3] citations become clickable chips (rewritten to #cite-n links). */
export function Answer({ text, onCite }: { text: string; onCite: (n: number) => void }) {
  const md = text.replace(/\[(\d+(?:\s*[,–-]\s*\d+)*)\]/g, (_, g: string) =>
    g
      .split(/\s*,\s*/)
      .map((part) => {
        const range = part.split(/\s*[–-]\s*/).map(Number);
        const nums = range.length === 2 ? Array.from({ length: range[1] - range[0] + 1 }, (_, i) => range[0] + i) : [range[0]];
        return nums.map((n) => `[${n}](#cite-${n})`).join("");
      })
      .join(""),
  );
  return (
    <div className="answer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            const m = href?.match(/^#cite-(\d+)$/);
            if (m) {
              return (
                <button className="cite" onClick={() => onCite(Number(m[1]))} title="Open source passage">
                  {children}
                </button>
              );
            }
            return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
          },
        }}
      >
        {md}
      </ReactMarkdown>
    </div>
  );
}
