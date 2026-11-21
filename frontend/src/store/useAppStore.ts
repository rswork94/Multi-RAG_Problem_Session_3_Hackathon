import { useSyncExternalStore } from "react";
import type { Area, Engine, Health, QueryResult } from "../types";

export interface PipelineToggles {
  disabled_nodes: string[];
  disabled_edges: string[];
}

export interface AppState {
  area: Area;
  engine: Engine | "compare";
  health: Health | null;
  selectedNodeIds: string[];
  blockedEdgeIds: string[];
  useSelectionScope: boolean;
  toggles: Record<Engine, PipelineToggles>;
  lastResult: Partial<Record<Engine, QueryResult>>;
  lastQuestion: string;
  kpiVersion: number; // bumps after each run so Tab 4 refetches
  graphVersion: number; // bumps after each ingest so Tab 2 refetches
}

const empty = (): PipelineToggles => ({ disabled_nodes: [], disabled_edges: [] });

function load(): Partial<AppState> {
  try {
    const raw = localStorage.getItem("multirag-ui");
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

const persisted = load();
let state: AppState = {
  area: (persisted.area as Area) ?? "tax",
  engine: (persisted.engine as AppState["engine"]) ?? "graph",
  health: null,
  selectedNodeIds: [],
  blockedEdgeIds: [],
  useSelectionScope: false,
  toggles: { standard: empty(), hyde: empty(), graph: empty() },
  lastResult: {},
  lastQuestion: (persisted.lastQuestion as string) ?? "",
  kpiVersion: 0,
  graphVersion: 0,
};
const listeners = new Set<() => void>();

export function setState(patch: Partial<AppState> | ((s: AppState) => Partial<AppState>)) {
  const p = typeof patch === "function" ? patch(state) : patch;
  state = { ...state, ...p };
  try {
    localStorage.setItem("multirag-ui", JSON.stringify({ area: state.area, engine: state.engine, lastQuestion: state.lastQuestion }));
  } catch {
    /* storage unavailable: per-viewer convenience only */
  }
  listeners.forEach((l) => l());
}

export function getState(): AppState {
  return state;
}

export function useApp<T>(sel: (s: AppState) => T): T {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => sel(state),
  );
}
