// Colour-blind-safe categorical palette (Okabe-Ito based) + engine colours shared by every chart.
// Fixed categorical order (never cycled): engines own slots 1-5. CSS vars switch steps for dark mode.
export const ENGINE_COLORS: Record<string, string> = {
  standard: "var(--series-1)",
  hyde: "var(--series-2)",
  graph: "var(--series-3)",
  graph_no_graph: "var(--series-4)",
  graph_no_rerank: "var(--series-5)",
};

// Latency breakdown segments use their own validated slot set (violet, green, magenta, yellow).
export const SEGMENTS = [
  { key: "retrieval_other_ms", label: "Retrieval (embed, dense, BM25, fusion, rerank)", color: "var(--seg-1)" },
  { key: "graph_ms", label: "Graph traversal", color: "var(--seg-2)" },
  { key: "llm_inference_ms", label: "LLM inference (HyDE + synthesis)", color: "var(--seg-3)" },
  { key: "guards_ms", label: "Guards", color: "var(--seg-4)" },
];

export const ENGINE_LABELS: Record<string, string> = {
  standard: "Standard RAG",
  hyde: "HyDE RAG",
  graph: "Graph RAG",
  graph_no_graph: "Graph − graph branch",
  graph_no_rerank: "Graph − rerank",
};

const PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#8c8c8c",
  "#7fc97f", "#beaed4", "#fdc086", "#386cb0", "#f0027f", "#bf5b17", "#1b9e77", "#d95f02"];

export const TYPE_COLORS: Record<string, string> = {
  Act: "#0072B2", Section: "#56B4E9", Rule: "#009E73", Notification: "#E69F00", Circular: "#D55E00",
  MasterDirection: "#CC79A7", Regulation: "#beaed4", Standard: "#7fc97f", Form: "#F0E442", Authority: "#386cb0",
  Rate: "#f0027f", Threshold: "#bf5b17", Deadline: "#1b9e77", Penalty: "#d62728", Control: "#17becf",
  Obligation: "#9467bd", Clause: "#8c564b", Party: "#e377c2", Agreement: "#bcbd22", Concept: "#8c8c8c",
  Chunk: "#b0b0b0", Document: "#606060",
};

export function typeColor(t: string): string {
  return TYPE_COLORS[t] ?? "#999";
}

export function paletteColor(i: number | null | undefined): string {
  return PALETTE[Math.abs(i ?? 0) % PALETTE.length];
}

export function kpiBand(value: number | null | undefined, green: number | null, amber: number | null, higherBetter: boolean) {
  if (value === null || value === undefined || green === null || amber === null) return "neutral";
  if (higherBetter) return value >= green ? "good" : value >= amber ? "warn" : "bad";
  return value <= green ? "good" : value <= amber ? "warn" : "bad";
}

export const AREA_LABELS: Record<string, string> = {
  tax: "Tax & Regulatory",
  grc: "Risk Advisory & GRC",
  deals: "Deal Advisory",
};
