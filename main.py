"""Multi-LLM chat app: fan out a prompt to ChatGPT, Claude, Grok, and Gemini,
then synthesize a summary with Claude. Streams results back via SSE so each
card fills in token-by-token. Supports multi-turn conversations — each
provider only sees its own prior responses."""

import asyncio
import inspect
import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import anthropic
from openai import AsyncOpenAI
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

load_dotenv(override=True)


def date_context() -> str:
    """Today's date + day of week, computed at request time so it's always
    fresh.  Injected into each provider's system prompt — LLMs have no
    internal clock and default to their training cutoff otherwise."""
    now = datetime.now()
    return now.strftime(f"Today's date is %A, %B {now.day}, %Y.")

_ANTHROPIC_DEFAULT = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
# Allow splitting answer vs. summary models — answer is called once per turn
# per request, summary reads all four answers so its input is bigger. A cheap
# model for answers and a smart one for summary is a useful cost split.
ANTHROPIC_ANSWER_MODEL = os.getenv("ANTHROPIC_ANSWER_MODEL", _ANTHROPIC_DEFAULT)
ANTHROPIC_SUMMARY_MODEL = os.getenv("ANTHROPIC_SUMMARY_MODEL", _ANTHROPIC_DEFAULT)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

anthropic_client = (
    anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if os.getenv("ANTHROPIC_API_KEY")
    else None
)
openai_client = (
    AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if os.getenv("OPENAI_API_KEY")
    else None
)
xai_client = (
    AsyncOpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1")
    if os.getenv("XAI_API_KEY")
    else None
)
gemini_client = (
    genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    if os.getenv("GEMINI_API_KEY")
    else None
)


SUMMARIZER_SYSTEM = (
    "You synthesize answers from four different LLMs (ChatGPT, Claude, Grok, "
    "Gemini) into a single concise response for the user. Identify points of "
    "agreement, note meaningful disagreements, and flag anything that looks "
    "wrong or unsupported. Be specific — cite which model said what when it "
    "matters. Keep the summary tight; do not pad with restatements. Format "
    "your response in Markdown."
)


# --------------------------------------------------------------------------- #
# Pricing.  $/1M tokens (input, output).  Update if a provider changes rates. #
# Cache discounts (esp. Anthropic) are ignored in v1 — costs are slight       #
# overestimates for repeated/cached prefixes.                                 #
# --------------------------------------------------------------------------- #


PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-5": (2.50, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Anthropic
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # xAI
    "grok-4": (3.00, 15.00),
    "grok-3": (3.00, 15.00),
    "grok-3-mini": (0.30, 0.50),
    # Google
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}


def compute_cost(
    model: str, input_tokens: int, output_tokens: int
) -> float | None:
    """Returns USD cost, or None if the model isn't in PRICING."""
    rates = PRICING.get(model)
    if rates is None:
        return None
    in_rate, out_rate = rates
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


# --------------------------------------------------------------------------- #
# Conversation message types (the wire protocol the frontend sends).          #
# --------------------------------------------------------------------------- #


class ConversationMessage(BaseModel):
    """One turn in the conversation.

    For user turns: role="user", content=<the prompt text>.
    For assistant turns: role="assistant", answers={ChatGPT: "...", ...}.
    """
    role: Literal["user", "assistant"]
    content: str | None = None  # for user turns
    answers: dict[str, str] | None = None  # for assistant turns


class AskRequest(BaseModel):
    messages: list[ConversationMessage]
    system: str | None = None  # optional per-conversation steering prompt


def build_system_text(user_system: str | None) -> str:
    """Combine the always-on date context with the optional user steering
    prompt for the per-provider system message."""
    parts = [date_context()]
    if user_system and user_system.strip():
        parts.append(user_system.strip())
    return "\n\n".join(parts)


def messages_for_provider(
    history: list[ConversationMessage], provider: str
) -> list[dict]:
    """Build an OpenAI-style messages list for one provider.

    Each provider sees its own prior responses only — never the other three's.
    Empty/missing prior responses become a placeholder so user/assistant
    alternation is preserved (Anthropic requires this).
    """
    out: list[dict] = []
    for m in history:
        if m.role == "user":
            out.append({"role": "user", "content": (m.content or "").strip()})
        elif m.role == "assistant":
            text = (m.answers or {}).get(provider) or "[no response]"
            out.append({"role": "assistant", "content": text})
    return out


# --------------------------------------------------------------------------- #
# Per-provider streaming functions. Each takes a messages list and yields     #
# text chunks.                                                                #
# --------------------------------------------------------------------------- #


async def stream_openai(
    messages: list[dict], usage_out: dict, user_system: str | None = None
) -> AsyncIterator[str]:
    if openai_client is None:
        yield "[OpenAI not configured: set OPENAI_API_KEY]"
        return
    stream = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": build_system_text(user_system)},
            *messages,
        ],
        stream=True,
        stream_options={"include_usage": True},  # final chunk carries usage
    )
    async for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        if chunk.usage:
            usage_out["input_tokens"] = chunk.usage.prompt_tokens
            usage_out["output_tokens"] = chunk.usage.completion_tokens


async def stream_anthropic(
    messages: list[dict], usage_out: dict, user_system: str | None = None
) -> AsyncIterator[str]:
    if anthropic_client is None:
        yield "[Anthropic not configured: set ANTHROPIC_API_KEY]"
        return
    async with anthropic_client.messages.stream(
        model=ANTHROPIC_ANSWER_MODEL,
        max_tokens=16000,
        system=build_system_text(user_system),
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text
        final = await stream.get_final_message()
        usage_out["input_tokens"] = final.usage.input_tokens
        usage_out["output_tokens"] = final.usage.output_tokens


async def stream_xai(
    messages: list[dict], usage_out: dict, user_system: str | None = None
) -> AsyncIterator[str]:
    if xai_client is None:
        yield "[xAI not configured: set XAI_API_KEY]"
        return
    stream = await xai_client.chat.completions.create(
        model=XAI_MODEL,
        messages=[
            {"role": "system", "content": build_system_text(user_system)},
            *messages,
        ],
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        if chunk.usage:
            usage_out["input_tokens"] = chunk.usage.prompt_tokens
            usage_out["output_tokens"] = chunk.usage.completion_tokens


async def stream_gemini(
    messages: list[dict], usage_out: dict, user_system: str | None = None
) -> AsyncIterator[str]:
    if gemini_client is None:
        yield "[Gemini not configured: set GEMINI_API_KEY]"
        return

    # Gemini expects role "model" instead of "assistant", and a different
    # content shape with `parts`.
    contents = [
        {
            "role": "model" if m["role"] == "assistant" else "user",
            "parts": [{"text": m["content"]}],
        }
        for m in messages
    ]

    # Retry up to 3× on transient 5xx — but only if no tokens have been
    # emitted yet (mid-stream errors can't be safely retried).
    last_err: Exception | None = None
    gemini_config = genai_types.GenerateContentConfig(
        system_instruction=build_system_text(user_system),
    )

    for attempt in range(3):
        emitted = False
        try:
            result = gemini_client.aio.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents,
                config=gemini_config,
            )
            if inspect.isawaitable(result):
                result = await result
            async for chunk in result:
                if chunk.text:
                    emitted = True
                    yield chunk.text
                # usage_metadata is cumulative; the final chunk has the totals.
                meta = getattr(chunk, "usage_metadata", None)
                if meta:
                    usage_out["input_tokens"] = meta.prompt_token_count or 0
                    usage_out["output_tokens"] = meta.candidates_token_count or 0
            return
        except genai_errors.ServerError as e:
            last_err = e
            if emitted:
                raise
            if attempt < 2:
                await asyncio.sleep(2**attempt)

    if last_err:
        raise last_err


async def stream_summary(
    prompt: str,
    answers: dict[str, str],
    usage_out: dict,
    user_system: str | None = None,
) -> AsyncIterator[str]:
    """Synthesize the four answers for the *current* turn. One-shot — no
    summary history, to keep token cost down."""
    if anthropic_client is None:
        yield "[Summary unavailable: ANTHROPIC_API_KEY not set]"
        return

    transcript = "\n\n".join(
        f"<{name}>\n{text}\n</{name}>" for name, text in answers.items()
    )
    # The user-set steering goes in the user message (not the cached system
    # block) so the SUMMARIZER_SYSTEM prefix cache stays valid.
    steering_block = ""
    if user_system and user_system.strip():
        steering_block = (
            f"User's conversation-level steering (apply when synthesizing):\n"
            f"{user_system.strip()}\n\n"
        )
    user_msg = (
        f"{date_context()}\n\n"
        f"{steering_block}"
        f"User's current prompt:\n{prompt}\n\n"
        f"Responses from each model:\n{transcript}\n\n"
        "Synthesize these into a single answer for the user."
    )

    async with anthropic_client.messages.stream(
        model=ANTHROPIC_SUMMARY_MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SUMMARIZER_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
        final = await stream.get_final_message()
        usage_out["input_tokens"] = final.usage.input_tokens
        usage_out["output_tokens"] = final.usage.output_tokens


# --------------------------------------------------------------------------- #
# SSE fan-out: 4 providers → shared queue → SSE event stream → client.        #
# --------------------------------------------------------------------------- #


PROVIDERS = [
    ("ChatGPT", stream_openai),
    ("Claude", stream_anthropic),
    ("Grok", stream_xai),
    ("Gemini", stream_gemini),
]


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# Which model is in play for each named slot (used for cost lookup).
def _model_for_slot(slot: str) -> str:
    return {
        "ChatGPT": OPENAI_MODEL,
        "Claude": ANTHROPIC_ANSWER_MODEL,
        "Grok": XAI_MODEL,
        "Gemini": GEMINI_MODEL,
        "Summary": ANTHROPIC_SUMMARY_MODEL,
    }.get(slot, "")


def _cost_payload(slot: str, usage: dict) -> dict:
    """Build the cost fields embedded in a done/summary_done event."""
    in_t = usage.get("input_tokens", 0) or 0
    out_t = usage.get("output_tokens", 0) or 0
    model = _model_for_slot(slot)
    cost = compute_cost(model, in_t, out_t)
    return {
        "model_id": model,  # actual model string (e.g. "claude-haiku-4-5")
        "input_tokens": in_t,
        "output_tokens": out_t,
        "cost": cost,  # may be None if model not in PRICING
    }


async def event_stream(
    history: list[ConversationMessage],
    user_system: str | None = None,
) -> AsyncIterator[str]:
    queue: asyncio.Queue = asyncio.Queue()
    answers: dict[str, str] = {name: "" for name, _ in PROVIDERS}

    latest_prompt = ""
    if history and history[-1].role == "user":
        latest_prompt = (history[-1].content or "").strip()

    # Aggregate the per-turn cost so we can emit a single total at the end.
    turn_total_cost = 0.0
    turn_has_unknown = False  # any slot used a model not in PRICING

    async def run_provider(name: str, stream_fn):
        nonlocal turn_total_cost, turn_has_unknown
        provider_messages = messages_for_provider(history, name)
        usage: dict = {}
        await queue.put({"type": "start", "model": name})
        try:
            async for chunk in stream_fn(provider_messages, usage, user_system):
                answers[name] += chunk
                await queue.put({"type": "delta", "model": name, "text": chunk})
            payload = _cost_payload(name, usage)
            if payload["cost"] is None:
                turn_has_unknown = True
            else:
                turn_total_cost += payload["cost"]
            await queue.put({"type": "done", "model": name, **payload})
        except Exception as e:
            err_text = f"[Error from {name}: {type(e).__name__}: {e}]"
            answers[name] = err_text
            await queue.put({"type": "error", "model": name, "error": err_text})

    async def driver():
        nonlocal turn_total_cost, turn_has_unknown
        tasks = [
            asyncio.create_task(run_provider(name, fn)) for name, fn in PROVIDERS
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        await queue.put({"type": "summary_start"})
        usage: dict = {}
        try:
            async for chunk in stream_summary(
                latest_prompt, answers, usage, user_system
            ):
                await queue.put({"type": "summary_delta", "text": chunk})
            payload = _cost_payload("Summary", usage)
            if payload["cost"] is None:
                turn_has_unknown = True
            else:
                turn_total_cost += payload["cost"]
            await queue.put({"type": "summary_done", **payload})
        except Exception as e:
            await queue.put(
                {
                    "type": "summary_error",
                    "error": f"[Summary failed: {type(e).__name__}: {e}]",
                }
            )

        # One final event with the rolled-up cost for this turn.
        await queue.put(
            {
                "type": "turn_total",
                "cost": turn_total_cost,
                "has_unknown": turn_has_unknown,
            }
        )

        await queue.put(None)

    driver_task = asyncio.create_task(driver())

    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield _sse(event)
    finally:
        if not driver_task.done():
            driver_task.cancel()


# --------------------------------------------------------------------------- #
# Single-slot retry: re-run one provider (or just the summary), then re-run   #
# the summary so it reflects the updated answer set.                          #
# --------------------------------------------------------------------------- #


PROVIDER_MAP = {name: fn for name, fn in PROVIDERS}


async def retry_event_stream(
    history: list[ConversationMessage],
    slot: str,
    current_answers: dict[str, str],
    user_system: str | None = None,
) -> AsyncIterator[str]:
    queue: asyncio.Queue = asyncio.Queue()
    answers: dict[str, str] = dict(current_answers)

    latest_prompt = ""
    if history and history[-1].role == "user":
        latest_prompt = (history[-1].content or "").strip()

    turn_total_cost = 0.0
    turn_has_unknown = False

    async def run_one(name: str, stream_fn):
        nonlocal turn_total_cost, turn_has_unknown
        provider_messages = messages_for_provider(history, name)
        usage: dict = {}
        answers[name] = ""  # reset; we're streaming fresh content
        await queue.put({"type": "start", "model": name})
        try:
            async for chunk in stream_fn(provider_messages, usage, user_system):
                answers[name] += chunk
                await queue.put({"type": "delta", "model": name, "text": chunk})
            payload = _cost_payload(name, usage)
            if payload["cost"] is None:
                turn_has_unknown = True
            else:
                turn_total_cost += payload["cost"]
            await queue.put({"type": "done", "model": name, **payload})
        except Exception as e:
            err_text = f"[Error from {name}: {type(e).__name__}: {e}]"
            answers[name] = err_text
            await queue.put({"type": "error", "model": name, "error": err_text})

    async def run_summary():
        nonlocal turn_total_cost, turn_has_unknown
        await queue.put({"type": "summary_start"})
        usage: dict = {}
        try:
            async for chunk in stream_summary(
                latest_prompt, answers, usage, user_system
            ):
                await queue.put({"type": "summary_delta", "text": chunk})
            payload = _cost_payload("Summary", usage)
            if payload["cost"] is None:
                turn_has_unknown = True
            else:
                turn_total_cost += payload["cost"]
            await queue.put({"type": "summary_done", **payload})
        except Exception as e:
            await queue.put(
                {
                    "type": "summary_error",
                    "error": f"[Summary failed: {type(e).__name__}: {e}]",
                }
            )

    async def driver():
        # If retrying a provider, run it first.  Always re-run the summary
        # (a fresh answer should ripple into the synthesis).
        if slot != "Summary":
            stream_fn = PROVIDER_MAP.get(slot)
            if stream_fn is not None:
                await run_one(slot, stream_fn)
        await run_summary()

        await queue.put(
            {
                "type": "turn_total",
                "cost": turn_total_cost,
                "has_unknown": turn_has_unknown,
            }
        )
        await queue.put(None)

    driver_task = asyncio.create_task(driver())

    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield _sse(event)
    finally:
        if not driver_task.done():
            driver_task.cancel()


class RetryRequest(BaseModel):
    messages: list[ConversationMessage]
    slot: Literal["ChatGPT", "Claude", "Grok", "Gemini", "Summary"]
    current_answers: dict[str, str]
    system: str | None = None


# --------------------------------------------------------------------------- #
# Per-IP rate limiting (sliding window).  In-memory — fine for a single       #
# process; resets on restart.  Set RATE_LIMIT_MAX to a high number locally    #
# if you want to disable it during dev.                                       #
# --------------------------------------------------------------------------- #


RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "5"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600"))
_RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    """Best-effort client IP. Behind Fly.io / most reverse proxies the real
    client IP is in X-Forwarded-For (first hop)."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    now = time.monotonic()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    bucket = _RATE_BUCKETS[ip]
    # Drop entries that fell out of the window.
    while bucket and bucket[0] < cutoff:
        bucket.popleft()

    if len(bucket) >= RATE_LIMIT_MAX:
        retry_after = max(int(bucket[0] + RATE_LIMIT_WINDOW_SECONDS - now) + 1, 1)
        minutes = (retry_after + 59) // 60
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit reached ({RATE_LIMIT_MAX} per "
                f"{RATE_LIMIT_WINDOW_SECONDS // 60} min). "
                f"Try again in ~{minutes} min."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    bucket.append(now)


app = FastAPI()


@app.get("/api/rate_limit")
def rate_limit_status(request: Request) -> dict:
    """Read-only view of the caller's current rate-limit state. Doesn't
    consume a slot."""
    ip = _client_ip(request)
    now = time.monotonic()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    bucket = _RATE_BUCKETS.get(ip)
    if not bucket:
        used = 0
        resets_in = 0
    else:
        # Don't mutate the global bucket here — count in-window entries.
        in_window = [t for t in bucket if t >= cutoff]
        used = len(in_window)
        resets_in = (
            int(in_window[0] + RATE_LIMIT_WINDOW_SECONDS - now) + 1
            if in_window
            else 0
        )

    return {
        "used": used,
        "remaining": max(RATE_LIMIT_MAX - used, 0),
        "limit": RATE_LIMIT_MAX,
        "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        "resets_in": resets_in,
    }


@app.post("/api/ask", dependencies=[Depends(rate_limit)])
async def ask(req: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        event_stream(req.messages, req.system),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/retry", dependencies=[Depends(rate_limit)])
async def retry(req: RetryRequest) -> StreamingResponse:
    return StreamingResponse(
        retry_event_stream(
            req.messages, req.slot, req.current_answers, req.system
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )
