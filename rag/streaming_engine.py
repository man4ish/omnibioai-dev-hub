import requests
import json
from typing import Generator, Dict, Any, Optional


class StreamingEngineV4:
    """
    Production-grade Ollama streaming engine for RAG V4 agent.

    Features:
    - Robust streaming parsing
    - Retry + fallback model
    - Agent-aware prompt injection
    - UI-friendly event streaming
    """

    def __init__(
        self,
        primary_model: str = "llama3",
        fallback_model: str = "mistral",
        base_url: str = "http://localhost:11434/api/chat",
        timeout: int = 120,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.url = base_url
        self.timeout = timeout

    # -----------------------------
    # PUBLIC STREAM INTERFACE
    # -----------------------------
    def stream(
        self,
        prompt: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> Generator[str, None, None]:

        meta = meta or {}

        # try primary model first
        yield from self._stream_with_model(self.primary_model, prompt, meta)

    # -----------------------------
    # INTERNAL STREAM CORE
    # -----------------------------
    def _stream_with_model(
        self,
        model: str,
        prompt: str,
        meta: Dict[str, Any]
    ):

        payload = {
            "model": model,
            "stream": True,
            "messages": [
                {
                    "role": "system",
                    "content": self._build_system_prompt(meta)
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                stream=True,
                timeout=self.timeout
            )

            if response.status_code != 200:
                yield f"[ERROR] Ollama HTTP {response.status_code}"
                yield from self._fallback(prompt, meta)
                return

            for line in response.iter_lines():

                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))

                    token = (
                        data.get("message", {})
                        .get("content", "")
                    )

                    if token:
                        yield token

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            yield f"[STREAM_ERROR]: {str(e)}"
            yield from self._fallback(prompt, meta)

    # -----------------------------
    # FALLBACK MODEL
    # -----------------------------
    def _fallback(self, prompt: str, meta: Dict[str, Any]):

        try:
            payload = {
                "model": self.fallback_model,
                "stream": True,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            response = requests.post(
                self.url,
                json=payload,
                stream=True,
                timeout=self.timeout
            )

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                except Exception:
                    continue

        except Exception as e:
            yield f"[FALLBACK_FAILED]: {str(e)}"

    # -----------------------------
    # SYSTEM PROMPT BUILDER (V4 AGENT AWARENESS)
    # -----------------------------
    def _build_system_prompt(self, meta: Dict[str, Any]) -> str:

        plan = meta.get("plan", {})

        return f"""
You are OmniBioAI RAG V4 Autonomous Agent.

You operate with:
- Hybrid retrieval (vector + graph + plugins)
- Memory-aware reasoning
- Tool-augmented context synthesis

Current execution plan:
{json.dumps(plan, indent=2)}

Rules:
- Use retrieved context if available
- Reason step-by-step internally
- Return concise scientific/technical answers
- If uncertain, say so clearly
"""