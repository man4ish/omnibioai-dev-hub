import { useAppStore } from "../store/appStore";

const API = "";

export async function streamRAG(query: string) {
  const setAnswer = useAppStore.getState().setAnswer;

  const res = await fetch(`${API}/rag/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  const reader = res.body?.getReader();
  const decoder = new TextDecoder();

  let fullText = "";

  while (true) {
    const { value, done } = await reader!.read();
    if (done) break;

    const chunk = decoder.decode(value);
    fullText += chunk;

    setAnswer(fullText);
  }

  return fullText;
}