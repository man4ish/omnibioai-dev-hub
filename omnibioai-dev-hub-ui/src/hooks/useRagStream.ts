import { useState } from "react";
import { ragStream } from "../api/client";

export function useRagStream() {
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const run = async (query: string) => {
    setAnswer("");
    setLoading(true);

    await ragStream(query, (token) => {
      setAnswer((prev) => prev + token);
    });

    setLoading(false);
  };

  return { answer, run, loading };
}