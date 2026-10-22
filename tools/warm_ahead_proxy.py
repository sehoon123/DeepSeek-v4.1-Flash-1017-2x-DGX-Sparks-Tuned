#!/usr/bin/env python3
"""An OpenAI / Anthropic API proxy in front of this recipe's server that warms the prompt cache ahead of a conversation's
next turn (optional; the server works the same without it). Replies pass through byte for byte; request bodies, headers
and replies never go to the log (it logs a client's hash, token counts and timings).

  LISTEN=0.0.0.0:8888 BACKEND=127.0.0.1:8000 KEY_FILE=~/.config/api-key python3 tools/warm_ahead_proxy.py

KEY_FILE (optional): requests must carry `Authorization: Bearer <its contents>` (401 otherwise); unset: no check.
WARM_AHEAD (1, the default; 0: a plain proxy), WARM_DELAY (seconds after a reply, 3), WARM_DELAY_TOOL (seconds after a
reply ending in tool calls, 0), WARM_STATE (the file of what it learned of each client, hashes only; ~/.cache).

The server's prompt cache keeps a prompt's state at 2,048-token chunk ends before its last 128 tokens, so a
conversation's next turn re-prefills everything after the last such end. After a chat reply (OpenAI
/v1/chat/completions or Anthropic /v1/messages), when no client request arrives for WARM_DELAY seconds, the proxy asks
the server's /tokenize how many tokens the conversation with this reply appended shares with any next request
(rendering only), and when that reaches a chunk end X the next request could resume and none does yet, sends the
server those shared tokens up to X + 128 as a token prompt (/v1/completions, max_tokens 1; the cache finds states by
tokens, whatever the route): the cache keeps the state at X, the server runs only 128 rows past it, and the next turn
resumes there, ~2,048 tokens later than before. X is at most shared - 128, so every next request (the shared tokens and
a message) resumes it: a deeper state would be useless to a short next message and would make the server's store free
the conversation's earlier state first. After a reply ending in tool calls the warm-ahead starts at once
(WARM_DELAY_TOOL, 0): the client runs the tools meanwhile, and the chunk computed is one its next request needs anyway. The state is the one the client's own request would compute (the same
tokens, the same chunk boundaries), so replies are the same bits (checked: a resumed reply equal to the same request
from an empty cache). A client request arriving while a warm-ahead runs closes it (the server stops at its current
chunk). The reply is appended as the server sent it; a client is warmed ahead only once its next request has shown that
it sends replies back that way, or in a way learned from it (reasoning dropped, tool-call arguments re-serialized, text
trimmed), per client (a hash of its system prompt and tools); a client that sends them back otherwise is not. No
harness is assumed.
"""
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading
import time

def _addr(v):
    host, _, port = v.rpartition(":")
    return (host or "127.0.0.1", int(port))


_kf = os.environ.get("KEY_FILE")
KEY = Path(os.path.expanduser(_kf)).read_text().strip() if _kf else None
LISTEN = _addr(os.environ.get("LISTEN", "0.0.0.0:8888"))
BACKEND = _addr(os.environ.get("BACKEND", "127.0.0.1:8000"))
MAX_BODY = 512 * 2**20      # the server takes up to 50 images and 4 videos (64 MB each) as base64 data URLs
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
               "transfer-encoding", "upgrade"}
WARM = os.environ.get("WARM_AHEAD", "1") != "0"
WARM_DELAY = float(os.environ.get("WARM_DELAY", "3"))
# after a reply that ends in tool calls the client runs them and comes back: the warm-ahead starts at once, the chunk it
# computes being one the next request needs anyway (a request arriving meanwhile waits at most for that chunk)
WARM_DELAY_TOOL = float(os.environ.get("WARM_DELAY_TOOL", "0"))
# what the proxy learned of each client (styles, which ones are followed), kept across restarts (hashes only)
STATE_FILE = os.environ.get("WARM_STATE", str(Path.home() / ".cache/warm-ahead-clients.json"))
CHUNK, WINDOW = 2048, 128   # the server's prompt chunk and the decoder's window (states kept at chunk ends <= len - 128)
CHAT, MESSAGES = "/v1/chat/completions", "/v1/messages"
NEXT_MIN = 3                # the fewest tokens a next request adds to the shared ones (a message's markers)


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


class ProxyServer(ThreadingHTTPServer):
    # ThreadingHTTPServer inherits TCPServer's tiny default listen backlog (5).
    # TensorFold serves up to 16 streams at once, so accept every benchmark/client connection promptly.
    request_queue_size = 128
    daemon_threads = True


# ------------------------------------------------------------------------------------------------ warm-ahead state
class Warm:
    def __init__(self):
        self.lock = threading.Lock()
        self.inflight = 0       # client requests being forwarded
        self.gen = 0            # bumped by every client request: a pending warm-ahead of an older reply gives up
        self.conn = None        # the backend connection of the warm-ahead in flight
        self.styles = {}        # client -> {"reasoning": bool, "args": "as-sent" | "compact", "strip": bool}
        self.expect = {}        # client -> (api, prefix length, prefix hash, the appended message as predicted)
        self.seen = set()       # clients whose echo matched the prediction (as learned): only these are warmed ahead
        self.given_up = set()   # clients whose echo differs in a way not learned: never warmed ahead
        self.stats = {"sent": 0, "done": 0, "stopped": 0, "echo_same": 0, "echo_learned": 0}


W = Warm()


def load_state():
    try:
        j = json.loads(Path(STATE_FILE).read_text())
        W.styles = {k: dict(v) for k, v in (j.get("styles") or {}).items()}
        W.seen = set(j.get("seen") or [])
        W.given_up = set(j.get("given_up") or [])
    except (OSError, ValueError, TypeError, AttributeError):
        pass


def save_state():
    try:
        tmp = STATE_FILE + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"styles": W.styles, "seen": sorted(W.seen), "given_up": sorted(W.given_up)}, f)
        os.replace(tmp, STATE_FILE)
    except OSError:
        pass


def jhash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def client_of(api, body):
    """A client's fingerprint: its system prompt and tools (stable for a harness and its configuration)."""
    msgs = body.get("messages") or []
    if api == CHAT:
        head = msgs[0] if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") in ("system", "developer") else None
        return jhash([head, body.get("tools")])[:12]
    return jhash([body.get("system"), body.get("tools")])[:12]


def compact(args):
    """Tool-call arguments re-serialized as JSON.stringify does (no spaces, non-ASCII kept)."""
    try:
        def fix(x):
            if isinstance(x, float) and x.is_integer() and abs(x) < 1e21:
                return int(x)
            if isinstance(x, dict):
                return {k: fix(v) for k, v in x.items()}
            if isinstance(x, list):
                return [fix(v) for v in x]
            return x
        return json.dumps(fix(json.loads(args)), separators=(",", ":"), ensure_ascii=False)
    except (ValueError, TypeError):
        return args


# ---------------------------------------------------------------------------------------- what a reply contained
class Reply:
    """A reply as it streams or arrives whole: the assistant turn it produced and its token counts."""

    def __init__(self, api):
        self.api = api
        self.content = ""
        self.reasoning = ""
        self.reasoning_field = "reasoning_content"
        self.calls = {}         # OpenAI: index -> {"id", "name", "arguments"}
        self.blocks = []        # Anthropic: content blocks in order
        self.finish = None
        self.prompt_tokens = None
        self.completion_tokens = None
        self.buf = b""
        self.broken = False

    def feed(self, data):   # SSE bytes as they pass
        self.buf += data
        while True:
            cut = self.buf.find(b"\n\n")
            if cut < 0:
                return
            event, self.buf = self.buf[:cut], self.buf[cut + 2:]
            for line in event.split(b"\n"):
                if line.startswith(b"data:"):
                    raw = line[5:].strip()
                    if raw and raw != b"[DONE]":
                        try:
                            self.event(json.loads(raw))
                        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                            self.broken = True

    def event(self, j):
        if self.api == CHAT:
            if j.get("usage"):
                self.prompt_tokens = j["usage"].get("prompt_tokens")
                self.completion_tokens = j["usage"].get("completion_tokens")
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    self.content += d["content"]
                for f in ("reasoning_content", "reasoning"):
                    if d.get(f):
                        self.reasoning += d[f]
                        self.reasoning_field = f
                for tc in d.get("tool_calls") or []:
                    c = self.calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    if tc.get("id"):
                        c["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        c["name"] += fn["name"]
                    if fn.get("arguments"):
                        c["arguments"] += fn["arguments"]
                if ch.get("finish_reason"):
                    self.finish = ch["finish_reason"]
            return
        t = j.get("type")
        if t == "message_start":
            u = (j.get("message") or {}).get("usage") or {}
            self.prompt_tokens = u.get("input_tokens")
        elif t == "content_block_start":
            b = dict(j.get("content_block") or {})
            if b.get("type") == "tool_use":
                b["_json"] = ""
            self.blocks.append(b)
        elif t == "content_block_delta" and self.blocks:
            b, d = self.blocks[-1], j.get("delta") or {}
            k = d.get("type")
            if k == "text_delta":
                b["text"] = b.get("text", "") + d.get("text", "")
            elif k == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + d.get("thinking", "")
            elif k == "signature_delta":
                b["signature"] = b.get("signature", "") + d.get("signature", "")
            elif k == "input_json_delta":
                b["_json"] = b.get("_json", "") + d.get("partial_json", "")
        elif t == "message_delta":
            self.finish = (j.get("delta") or {}).get("stop_reason") or self.finish
            u = j.get("usage") or {}
            if u.get("output_tokens") is not None:
                self.completion_tokens = u["output_tokens"]

    def whole(self, j):     # a reply that is not a stream
        if self.api == CHAT:
            ch = (j.get("choices") or [{}])[0]
            m = ch.get("message") or {}
            self.content = m.get("content") or ""
            for f in ("reasoning_content", "reasoning"):
                if m.get(f):
                    self.reasoning, self.reasoning_field = m[f], f
            for i, tc in enumerate(m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                self.calls[i] = {"id": tc.get("id"), "name": fn.get("name", ""), "arguments": fn.get("arguments", "")}
            self.finish = ch.get("finish_reason")
            u = j.get("usage") or {}
        else:
            self.blocks = [dict(b) for b in j.get("content") or []]
            self.finish = j.get("stop_reason")
            u = j.get("usage") or {}
            u = {"prompt_tokens": u.get("input_tokens"), "completion_tokens": u.get("output_tokens")}
        self.prompt_tokens = u.get("prompt_tokens")
        self.completion_tokens = u.get("completion_tokens")

    def ended_turn(self):
        # a reply cut at max_tokens is sent back as it is by most clients too
        ok = ("stop", "tool_calls", "length") if self.api == CHAT else ("end_turn", "tool_use", "stop_sequence", "max_tokens")
        return not self.broken and self.finish in ok

    def message(self, style):
        """The assistant message a client sends back, as the server's reply has it (and as `style` says this client
        changes it)."""
        if self.api == CHAT:
            text = self.content.strip() if style.get("strip") else self.content
            m = {"role": "assistant", "content": text if text.strip() else None}
            if self.reasoning.strip() and style.get("reasoning", True):
                m[self.reasoning_field] = self.reasoning
            if self.calls:
                m["tool_calls"] = []
                for i in sorted(self.calls):
                    c = self.calls[i]
                    args = compact(c["arguments"]) if style.get("args") == "compact" else c["arguments"]
                    m["tool_calls"].append({"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": args}})
            return m
        blocks = []
        for b in self.blocks:
            b = {k: v for k, v in b.items() if not k.startswith("_")}
            if b.get("type") == "tool_use" and "_json" in self.blocks[len(blocks)]:
                raw = self.blocks[len(blocks)]["_json"]
                try:
                    b["input"] = json.loads(raw) if raw else (b.get("input") or {})
                except ValueError:
                    pass
            if b.get("type") in ("thinking", "redacted_thinking") and not style.get("reasoning", True):
                blocks.append(None)
                continue
            if b.get("type") == "text" and style.get("strip"):
                b["text"] = b.get("text", "").strip()
            blocks.append(b)
        return {"role": "assistant", "content": [b for b in blocks if b is not None]}


# --------------------------------------------------------------------------------------------- learning the echo
def learn(api, body, who):
    """A client request after a reply of the same client: how it sent that reply back."""
    exp = W.expect.get(who)
    msgs = body.get("messages") or []
    if not exp or exp[0] != api or len(msgs) <= exp[1] or jhash(msgs[:exp[1]]) != exp[2]:
        return
    echo, ours, raw = msgs[exp[1]], exp[3], exp[4]
    if not isinstance(echo, dict) or echo.get("role") != "assistant":
        return
    if jhash(echo) == jhash(ours):
        W.stats["echo_same"] += 1
        if who not in W.seen:
            W.seen.add(who)
            save_state()
        return
    style = dict(W.styles.get(who, {}))
    before = dict(style)
    if api == CHAT:
        has = bool((echo.get("reasoning_content") or echo.get("reasoning") or "").strip())
        if raw.reasoning.strip():
            style["reasoning"] = has
        if raw.calls and echo.get("tool_calls"):
            sent = [((t.get("function") or {}).get("arguments") or "") for t in echo["tool_calls"]]
            mine = [raw.calls[i]["arguments"] for i in sorted(raw.calls)]
            if sent != mine and sent == [compact(a) for a in mine]:
                style["args"] = "compact"
            elif sent == mine:
                style["args"] = "as-sent"
        txt = echo.get("content") if isinstance(echo.get("content"), str) else None
        if txt is not None and raw.content and txt == raw.content.strip() and txt != raw.content:
            style["strip"] = True
    else:
        content = echo.get("content") if isinstance(echo.get("content"), list) else []
        if any(b.get("type") in ("thinking", "redacted_thinking") for b in raw.blocks):
            style["reasoning"] = any(isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking") for b in content)
    if style != before:
        W.styles[who] = style
        W.stats["echo_learned"] += 1
    # the learned style must explain the whole echo, else this client is not warmed ahead (a wrong prediction would
    # keep a state its next request cannot use)
    if jhash(echo) == jhash(raw.message(style)):
        W.seen.add(who)
        save_state()
        log(f"warm-ahead: client {who} echoes replies as {style or 'sent'}")
    else:
        W.seen.discard(who)
        W.given_up.add(who)
        save_state()
        mine = raw.message(style)
        diff = {k: (len(json.dumps(mine.get(k), ensure_ascii=False)), len(json.dumps(echo.get(k), ensure_ascii=False)))
                for k in sorted(set(mine) | set(echo)) if mine.get(k) != echo.get(k)}
        log(f"warm-ahead: client {who} echoes replies in a way not learned (fields, lengths predicted/sent: {diff}): "
            "no warm-ahead for it")


def tokens(path, base, msgs):
    """The ids the server would run for these messages (its /tokenize: rendering only, no GPU work)."""
    b = {k: v for k, v in base.items() if k not in ("stream", "stream_options", "max_tokens", "max_completion_tokens", "n")}
    b["messages"] = msgs
    c = http.client.HTTPConnection(*BACKEND, timeout=60)
    try:
        c.request("POST", "/tokenize", body=json.dumps(b, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        return json.loads(data)["tokens"] if r.status == 200 else None
    except (OSError, http.client.HTTPException, ValueError, KeyError):
        return None
    finally:
        c.close()


def plan(api, path, base, msgs, m, prompt_tokens, completion_tokens):
    """The warm-ahead request (bytes) and a note, or (None, why). It keeps the state at the deepest chunk end X of the
    tokens the next request shares (the conversation and this reply): the request is those messages and a filler user
    message long enough to put X before the last 128 tokens (states are kept only there), so the filler never reaches
    the state kept."""
    kept = (prompt_tokens - WINDOW) // CHUNK * CHUNK if prompt_tokens else 0
    if api == CHAT:
        a = tokens(path, base, msgs + [m])
        b = tokens(path, base, msgs + [m, {"role": "user", "content": "x"}])
        if a is None or b is None:
            return None, "no token count", None
        shared = 0
        while shared < min(len(a), len(b)) and a[shared] == b[shared]:
            shared += 1
        # the shared tokens' ids up to a chunk end X and the window past it, as a token prompt (/v1/completions; the
        # cache finds states by tokens, whatever the route): the server keeps X and runs only 128 rows past it. X is
        # at most shared - 128, so the prompt stays inside the shared tokens and every next request (shared + a message)
        # resumes it (a deeper state it could not use would also make the store free this conversation's earlier one)
        X = (shared - WINDOW) // CHUNK * CHUNK
        if X <= kept:
            return None, f"shares {shared} tokens: no chunk end past {kept}", None
        warm = {"model": base.get("model") or "model", "prompt": a[:X + WINDOW], "max_tokens": 1, "temperature": 0,
                "priority": "background"}
        return json.dumps(warm).encode(), f"shares {shared} tokens, state {kept}->{X}", "/v1/completions"
    else:
        # Anthropic: no ids to compare; only when the reply clearly crossed a chunk end past the window
        if prompt_tokens is None or completion_tokens is None:
            return None, "no usage", None
        X = (prompt_tokens + completion_tokens + NEXT_MIN - WINDOW) // CHUNK * CHUNK
        if X <= kept:
            return None, "no chunk end crossed past the window", None
        note = f"prompt {prompt_tokens}+{completion_tokens}, state {kept}->{X}"
        tail = [m, {"role": "user", "content": "x"}]
    warm = {k: v for k, v in base.items() if k not in ("stream", "stream_options", "n", "logprobs", "top_logprobs")}
    warm["messages"] = msgs + tail
    warm["max_tokens"] = 1
    if "max_completion_tokens" in warm:
        warm["max_completion_tokens"] = 1
    return json.dumps(warm, ensure_ascii=False).encode(), note, path


def schedule(api, path, body, who, reply):
    """After a reply: the warm-ahead of this conversation, WARM_DELAY seconds later if nothing else came."""
    msgs = body.get("messages") or []
    style = W.styles.get(who, {})
    m = reply.message(style)
    if api == CHAT and m.get("content") is None and not m.get("tool_calls"):
        return      # a client drops an empty reply: nothing to append
    W.expect[who] = (api, len(msgs), jhash(msgs), m, reply)
    if len(W.expect) > 64:
        W.expect.pop(next(iter(W.expect)))
    if reply.prompt_tokens is not None and reply.prompt_tokens + (reply.completion_tokens or 0) < CHUNK:
        return      # a short conversation: no chunk end to keep
    if who not in W.seen or who in W.given_up:
        return      # how this client sends replies back is not known yet (or not followed)
    with W.lock:
        gen = W.gen

    def idle():
        with W.lock:
            return W.gen == gen and W.inflight == 0

    tool_turn = reply.finish in ("tool_calls", "tool_use")
    delay = WARM_DELAY_TOOL if tool_turn else WARM_DELAY

    def later():
        if delay > 0:
            time.sleep(delay)
        if not idle():
            return
        data, note, route = plan(api, path, body, msgs, m, reply.prompt_tokens, reply.completion_tokens)
        if data is None:
            log(f"warm-ahead: client {who}: none ({note})")
            return
        with W.lock:
            if W.gen != gen or W.inflight > 0:
                return
            conn = http.client.HTTPConnection(*BACKEND, timeout=900)
            W.conn = conn
            W.stats["sent"] += 1
        t0 = time.time()
        ok = False
        try:
            conn.request("POST", route, body=data, headers={"Content-Type": "application/json"})
            rep = conn.getresponse()
            rep.read()
            ok = rep.status == 200
        except (OSError, http.client.HTTPException):
            pass
        finally:
            with W.lock:
                if W.conn is conn:
                    W.conn = None
                W.stats["done" if ok else "stopped"] += 1
            conn.close()
        log(f"warm-ahead: client {who} {note}{' (tool turn)' if tool_turn else ''}: {'done' if ok else 'stopped'} in {time.time() - t0:.1f} s; {W.stats}")

    threading.Thread(target=later, daemon=True).start()


def client_arrived():
    """A client request: no warm-ahead runs beside it."""
    with W.lock:
        W.inflight += 1
        W.gen += 1
        conn, W.conn = W.conn, None
    if conn is not None:
        try:
            conn.sock.shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass
        conn.close()


def client_left():
    with W.lock:
        W.inflight -= 1


# ------------------------------------------------------------------------------------------------------- the proxy
class API(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "WarmAheadProxy"
    sys_version = ""

    def log_message(self, *args):
        return

    def error(self, code, message):
        data = json.dumps({"error": {"message": message, "type": "api_error", "code": code}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)
        self.close_connection = True

    def forward(self):
        self.connection.settimeout(120)
        if KEY is not None and not hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + KEY).encode()):
            self.error(401, "A valid API key is required")
            return
        if self.headers.get("Transfer-Encoding"):
            self.error(400, "Use Content-Length for request bodies")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.error(400, "Invalid Content-Length")
            return
        if size < 0 or size > MAX_BODY:
            self.error(413, "Request body exceeds the API limit")
            return
        body = self.rfile.read(size) if size else None
        if body is not None and len(body) != size:
            self.error(400, "Incomplete request body")
            return
        # a chat request this proxy may warm ahead of: its body, its client, and its reply as it passes
        api = parsed = who = reply = None
        route = self.path.split("?")[0]
        if WARM and self.command == "POST" and route in (CHAT, MESSAGES) and body:
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list) and parsed.get("n", 1) == 1:
                    api = route
                    who = client_of(api, parsed)
                    learn(api, parsed, who)
                    reply = Reply(api)
            except (ValueError, TypeError, AttributeError, KeyError):
                api = None
        client_arrived()
        excluded = HOP_HEADERS | {"authorization", "host", "content-length"}
        excluded.update(x.strip().lower() for x in self.headers.get("Connection", "").split(",") if x.strip())
        headers = {k: v for k, v in self.headers.items() if k.lower() not in excluded}
        backend = http.client.HTTPConnection(*BACKEND, timeout=3600)   # a 1M-token prefill takes ~16 minutes
        started = completed = False
        whole = b""
        try:
            backend.request(self.command, self.path, body=body, headers=headers)
            rep = backend.getresponse()
            self.send_response(rep.status)
            skip = HOP_HEADERS | {"server", "date"}
            skip.update(x.strip().lower() for x in (rep.getheader("Connection") or "").split(",") if x.strip())
            for name, value in rep.getheaders():
                if name.lower() not in skip:
                    self.send_header(name, value)
            # http.client decodes chunked replies; closing delimits a reply without a length, and read1 passes SSE
            # events on as they arrive instead of buffering the stream
            self.send_header("Connection", "close")
            self.end_headers()
            started = True
            streaming = "text/event-stream" in (rep.getheader("Content-Type") or "")
            if self.command != "HEAD":
                while chunk := rep.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    if reply is not None:
                        if streaming:
                            reply.feed(chunk)
                        elif len(whole) < 64 * 2**20:
                            whole += chunk
            completed = rep.status == 200
            if reply is not None and completed and not streaming:
                try:
                    reply.whole(json.loads(whole))
                except (ValueError, TypeError, AttributeError):
                    reply.broken = True
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except (OSError, http.client.HTTPException):
            if not started:
                self.error(502, "The model API is unavailable (starting or stopped)")
        finally:
            backend.close()
            self.close_connection = True
            client_left()
            if reply is not None and completed and reply.ended_turn():
                try:
                    schedule(api, self.path, parsed, who, reply)
                except Exception as e:      # never let the warm-ahead disturb the proxy
                    log(f"warm-ahead: not scheduled ({type(e).__name__})")

    do_GET = do_POST = do_HEAD = do_OPTIONS = do_DELETE = forward


if __name__ == "__main__":
    load_state()
    server = ProxyServer(LISTEN, API)
    print(f"warm-ahead proxy ({'key checked' if KEY else 'no key'}) on {LISTEN[0]}:{LISTEN[1]} -> {BACKEND[0]}:{BACKEND[1]}; warm-ahead "
          f"{'on, ' + str(WARM_DELAY) + ' s after a reply (' + str(WARM_DELAY_TOOL) + ' s after tool calls), ' + str(len(W.seen)) + ' clients known' if WARM else 'off'}", flush=True)
    server.serve_forever()
