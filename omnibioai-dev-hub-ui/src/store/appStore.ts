import { create } from "zustand";

interface AppState {
  query: string;
  answer: string;
  trace: any;
  graph: any;
  vector: any;
  setQuery: (q: string) => void;
  setAnswer: (a: string) => void;
  setTrace: (t: any) => void;
}

export const useAppStore = create<AppState>((set) => ({
  query: "",
  answer: "",
  trace: null,
  graph: null,
  vector: null,

  setQuery: (q) => set({ query: q }),
  setAnswer: (a) => set({ answer: a }),
  setTrace: (t) => set({ trace: t }),
}));