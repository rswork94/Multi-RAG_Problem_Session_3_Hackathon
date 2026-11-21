export type Area = "tax" | "grc" | "deals";
export type Engine = "standard" | "hyde" | "graph";

export interface TraceEntry {
  step: string;
  label: string;
  status: "running" | "done" | "error" | "bypassed";
  duration_ms?: number;
  stats?: Record<string, any>;
  error?: string;
}

export interface Passage {
  n: number;
  chunk_id: string;
  text: string;
  source: string;
  title: string;
  doc_type: string;
  issued_date: string;
  section: string;
  page: number;
  practice_area: Area;
  n_tokens: number;
  injection_flag: boolean;
  ranks?: Record<string, number>;
  hop?: number | null;
  rerank?: number | null;
  rrf?: number | null;
  cosine?: number | null;
}

export interface Latency {
  e2e_ms: number;
  retrieval_ms: number;
  graph_ms: number;
  llm_pre_ms: number;
  synthesis_ms: number;
  guards_ms: number;
  retrieval_other_ms: number;
  llm_inference_ms: number;
  embed_ms?: number;
  rerank_ms?: number;
  ttft_ms?: number;
}

export interface Kpis {
  rcr: number | null;
  rcr_method?: string;
  rcr_facts?: { fact: string; present: boolean }[];
  faithfulness: number | null;
  faithfulness_method?: string;
  abstained?: boolean;
  hallucination_rate?: number | null;
  unsupported_claims?: string[];
  er_accuracy: number | null;
  entity_link_hit?: number | null;
  latency_ms: number;
  latency: Latency;
  guard_compliant?: boolean;
  citation_coverage?: number;
  answer_check?: { missing: string[]; forbidden_present: string[]; pass: boolean } | null;
  refused?: boolean;
}

export interface Triple {
  src: string;
  dst: string;
  rel: string;
  hop: number;
  src_label: string;
  dst_label: string;
  refs?: number[];
}

export interface QueryResult {
  engine: Engine;
  engine_label: string;
  area: Area;
  question: string;
  masked_question: string;
  answer: string;
  citations: { n: number; chunk_id: string; source: string; section: string; page: number }[];
  context: Passage[];
  triples: Triple[];
  linked_entities: { id: string; label: string; type: string; how: string }[];
  hypotheses: string[];
  sufficiency?: Record<string, any> | null;
  hop_k_used?: number;
  loops: number;
  guard_report: Record<string, any>;
  kpis: Kpis;
  trace: TraceEntry[];
  usage: Record<string, number>;
  mode: string;
  model_used: string;
  fallback_used: boolean;
  pipeline?: { disabled_nodes: string[]; disabled_edges: string[]; effective: string[]; hash: string };
  hyperparams: Hyperparams;
  gold_id?: string | null;
  gold_category?: string | null;
  selected_nodes: number;
  llm_mode: string;
}

export interface Hyperparams {
  chunk_size: number;
  chunk_overlap: number;
  hyde_n: number;
  hop_k: number;
  rrf_alpha: number;
}

export interface HpMeta {
  group: string;
  min: number;
  max: number;
  step: number;
  reingest: boolean;
  help: string;
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  community: number;
  n2v_cluster?: number | null;
  pagerank: number;
  degree: number;
  area: string;
  aliases: string;
  mentions: number;
  x?: number;
  y?: number;
  z?: number;
  fx?: number;
  fy?: number;
  fz?: number;
}

export interface GraphLink {
  id: string;
  source: string | GraphNode;
  target: string | GraphNode;
  rel: string;
  weight: number;
  evidence: string;
}

export interface Health {
  llm_mode: "online" | "offline";
  provider?: string;
  synth_effort?: string;
  synth_model: string;
  utility_model: string;
  embed_model: string;
  rerank_model: string;
  device: string;
  warm: { status: string; seconds?: number };
  documents: number;
  chunks: number;
  graph_nodes: number;
  corpus_version: string | null;
  gold_questions: number;
  reingest_required: boolean;
}

export interface GoldQuestion {
  id: string;
  area: Area;
  category: string;
  question: string;
  hops: number;
}

export interface RunRow {
  at: number;
  engine: Engine;
  area: Area;
  question: string;
  gold_id?: string;
  category?: string;
  mode: string;
  model: string;
  pipeline_hash: string;
  disabled: string[];
  hyperparams: Hyperparams;
  kpis: Kpis;
  refused: boolean;
  cost_usd: number;
}

export type SSEEvent =
  | { type: "step"; step: string; label: string; status: string; duration_ms?: number; stats?: any; error?: string; engine?: Engine }
  | { type: "progress"; step: string; done: number; total: number }
  | { type: "token"; text: string; engine?: Engine }
  | { type: "replace"; text: string; engine?: Engine }
  | { type: "bench"; phase: string; done: number; total: number; question?: string; variant?: string }
  | { type: "error"; message: string }
  | ({ type: "final" } & Record<string, any>);
