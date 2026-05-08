import requests
import json
from typing import Generator, Dict, Any, Optional


# =========================================================
# STREAMING ENGINE V4 - OMNIBIOAI
# Agentic + Robust + UI-friendly + Fault-tolerant
# =========================================================

class StreamingEngineV4:
    """
    Production-grade Ollama streaming engine for RAG V4.

    Features:
    - Primary + fallback model routing
    - Structured SSE-ready output
    - Safe JSON parsing
    - Agent-aware system prompt injection
    - Retry + graceful degradation
    """

    def __init__(
        self,
        primary_model: str = "llama3",
        fallback_model: str = "mistral",
        base_url: str = "http://localhost:11434/api/chat",
        timeout: int = 120,
        max_retries: int = 1,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    # =========================================================
    # PUBLIC STREAM INTERFACE
    # =========================================================
    def stream(
        self,
        prompt: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> Generator[Dict[str, Any], None, None]:

        meta = meta or {}

        # try primary model first
        yield from self._stream_with_model(self.primary_model, prompt, meta)

    # =========================================================
    # CORE STREAM HANDLER
    # =========================================================
    def _stream_with_model(
        self,
        model: str,
        prompt: str,
        meta: Dict[str, Any]
    ) -> Generator[Dict[str, Any], None, None]:

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
                yield {"type": "error", "content": f"Ollama HTTP {response.status_code}"}
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
                        yield {
                            "type": "token",
                            "content": token,
                            "model": model
                        }

                except json.JSONDecodeError:
                    continue

            # end signal
            yield {"type": "done", "model": model}

        except Exception as e:
            yield {"type": "error", "content": str(e)}
            yield from self._fallback(prompt, meta)

    # =========================================================
    # FALLBACK MODEL STREAM
    # =========================================================
    def _fallback(
        self,
        prompt: str,
        meta: Dict[str, Any]
    ) -> Generator[Dict[str, Any], None, None]:

        payload = {
            "model": self.fallback_model,
            "stream": True,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
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
                        yield {
                            "type": "token",
                            "content": token,
                            "model": self.fallback_model
                        }

                except Exception:
                    continue

            yield {"type": "done", "model": self.fallback_model}

        except Exception as e:
            yield {
                "type": "fatal",
                "content": f"Fallback failed: {str(e)}"
            }

    # =========================================================
    # SYSTEM PROMPT (AGENTIC V4)
    # =========================================================
    def _build_system_prompt(self, meta: Dict[str, Any]) -> str:

        plan = meta.get("plan", {})
        memory = meta.get("memory", "")

        return f"""
You are OmniBioAI RAG V4 Autonomous Agent.

Capabilities:
- Hybrid retrieval (vector + graph + plugin systems)
- Memory-aware reasoning
- Tool-augmented scientific analysis
- Multi-step reasoning with context fusion

Execution Plan:
{json.dumps(plan, indent=2)}

Memory Context:
{memory}

Rules:
- Use provided context if available
- Prefer structured reasoning
- Be precise and technical
- If uncertain, explicitly say so
- Avoid hallucination
"""