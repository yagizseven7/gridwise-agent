"""A local HTTP server that speaks the Azure OpenAI and Anthropic wire formats.

It lets the provider tests drive the *real* SDKs (`openai` via `langchain-openai`,
`anthropic` via `langchain-anthropic`) end to end -- request serialisation, auth
headers, URL layout, response parsing, token usage, error mapping -- with no
network and no keys. It is a test double for the HTTP API, not for the SDKs.

Routes:
    POST /openai/v1/chat/completions                       Azure v1 GA API
    POST /openai/v1/embeddings
    POST /openai/deployments/<dep>/chat/completions        Azure classic API (?api-version=)
    POST /openai/deployments/<dep>/embeddings
    POST /v1/messages                                       Anthropic Messages API

Replies are chosen from the system prompt (router / writer / verifier), so the
whole LangGraph pipeline can run through it. Behaviour switches:
    question contains FILTER_ME   -> Azure 400 content_filter (Prompt Shields style)
    question contains FILTER_OUT  -> 200 with finish_reason "content_filter"
    question contains RATE_LIMIT  -> 429 on every call
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PLAN = {"objective": "cold spell rule", "steps": [
    {"tool": "search_playbook", "args": {"query": "cold spell protocol", "k": 2}, "why": "rules"}]}
ANSWER = {"answer": "During a cold spell the peak shaving trigger drops from 92 percent to 88 percent "
                    "of contracted capacity. Sources: site_operations_playbook.md#Cold spell protocol.",
          "confidence": "high", "flags": []}


def _verdict(prompt: str) -> dict[str, Any]:
    answer = re.search(r"<answer>(.*?)</answer>", prompt, re.S)
    body = (answer.group(1) if answer else "").split("Sources:")[0].strip()
    return {"claims": [{"claim": body, "supported": True, "source": "E1"}]}


def _reply_for(system: str, user: str) -> str:
    if "router agent" in system:
        return json.dumps(PLAN)
    if "verification agent" in system:
        return json.dumps(_verdict(user))
    return json.dumps(ANSWER)


def _embedding(text: str, dims: int = 64) -> list[float]:
    """Deterministic bag-of-words hash vector, L2-normalised."""
    vec = [0.0] * dims
    for word in re.findall(r"[a-z]+", text.lower()):
        if len(word) > 3:
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


class FakeProviderServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # keep test output quiet
                pass

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                server.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                        "json": payload})
                path = self.path.split("?")[0]
                if path.endswith("/embeddings"):
                    return self._send(200, server.embeddings(payload))
                if path.endswith("/chat/completions"):
                    return server.openai_chat(self, payload)
                if path == "/v1/messages":
                    return server.anthropic_messages(self, payload)
                self._send(404, {"error": {"message": f"no route {path}"}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "FakeProviderServer":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    # --- OpenAI / Azure -------------------------------------------------------
    def openai_chat(self, handler: Any, payload: dict[str, Any]) -> None:
        messages = payload.get("messages", [])
        system = next((m["content"] for m in messages if m["role"] in ("system", "developer")), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "RATE_LIMIT" in user:
            return handler._send(429, {"error": {"code": "429", "message": "Rate limit is exceeded. Try again later."}})
        if "FILTER_ME" in user:
            return handler._send(400, {"error": {
                "message": "The response was filtered due to the prompt triggering Azure OpenAI's content "
                           "management policy.", "type": None, "param": "prompt", "code": "content_filter",
                "status": 400, "innererror": {"code": "ResponsibleAIPolicyViolation",
                                              "content_filter_result": {"jailbreak": {"filtered": True, "detected": True}}}}})
        finish = "content_filter" if "FILTER_OUT" in user and "router agent" not in system else "stop"
        content = "" if finish == "content_filter" else _reply_for(system, user)
        handler._send(200, {
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": payload.get("model", "dep"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        })

    def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        texts = payload["input"] if isinstance(payload["input"], list) else [payload["input"]]
        data = []
        for i, text in enumerate(texts):
            vector = _embedding(str(text))
            if payload.get("encoding_format") == "base64":
                encoded: Any = base64.b64encode(struct.pack(f"{len(vector)}f", *vector)).decode()
            else:
                encoded = vector
            data.append({"object": "embedding", "index": i, "embedding": encoded})
        return {"object": "list", "data": data, "model": payload.get("model", "emb"),
                "usage": {"prompt_tokens": 5 * len(texts), "total_tokens": 5 * len(texts)}}

    # --- Anthropic -----------------------------------------------------------------
    def anthropic_messages(self, handler: Any, payload: dict[str, Any]) -> None:
        system = payload.get("system", "")
        if isinstance(system, list):
            system = " ".join(block.get("text", "") for block in system)
        first = payload.get("messages", [{}])[0].get("content", "")
        user = first if isinstance(first, str) else " ".join(b.get("text", "") for b in first)
        handler._send(200, {
            "id": "msg_test", "type": "message", "role": "assistant", "model": payload.get("model", "claude"),
            "content": [{"type": "text", "text": _reply_for(system, user)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 110, "output_tokens": 25},
        })
