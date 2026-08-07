#!/usr/bin/env python3
"""Benchmark one warmed decode regime against an OpenAI-compatible SGLang server."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
import math
import re
import statistics
import time
from typing import Any

import httpx
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


@dataclass
class StreamResult:
    ttft_s: float = 0.0
    total_time_s: float = 0.0
    total_tokens: int = 0
    decode_time_s: float = 0.0
    decode_tokens: int = 0
    first_token_at_s: float = 0.0
    end_at_s: float = 0.0
    error: str | None = None


@dataclass
class RegimeSummary:
    label: str
    base_url: str
    model: str
    concurrency: int
    context_tokens_target: int
    context_tokens_actual: int
    max_tokens: int
    successful_requests: int
    failed_requests: int
    total_completion_tokens: int
    median_ttft_s: float
    p95_ttft_s: float
    median_request_time_s: float
    median_request_decode_tps: float
    aggregate_client_decode_tps: float
    median_server_decode_tps: float
    median_server_running_reqs: float
    median_server_queue_reqs: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key, value_str = parts[0], parts[-1]
        try:
            value = float(value_str)
        except ValueError:
            continue
        if "tp_rank=" in key and 'tp_rank="0"' not in key:
            continue
        metrics[key] = value
    return metrics


def _extract_metric(metrics: dict[str, float], name: str) -> float:
    for key, value in metrics.items():
        if key.startswith(name):
            return value
    return 0.0


def _token_count(tokenized: Any) -> int:
    if isinstance(tokenized, BatchEncoding):
        input_ids = tokenized.get("input_ids")
        if input_ids is None:
            return 0
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
    if isinstance(tokenized, list):
        if tokenized and isinstance(tokenized[0], list):
            return len(tokenized[0])
        return len(tokenized)
    return len(tokenized)


async def _wait_until_ready(base_url: str, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
        last_error = "server not ready"
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{base_url}/v1/models")
                if resp.status_code == 200:
                    payload = resp.json()
                    model_id = payload["data"][0]["id"]
                    return model_id
                last_error = f"HTTP {resp.status_code}"
            except Exception as exc:  # pragma: no cover - diagnostic path
                last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(1.0)
    raise RuntimeError(f"server at {base_url} did not become ready: {last_error}")


def _make_tokenizer(path: str):
    return AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
    )


def _build_messages_for_context(tokenizer, target_tokens: int) -> tuple[list[dict[str, str]], int]:
    prefix = (
        "You will be given filler context. Ignore the filler and later continue a long numbered list. "
        "The filler exists only to control prompt length.\n\n"
    )
    suffix = (
        "\n\nIgnore the filler above. Respond with a long sequence of integers separated by single spaces, "
        "starting from 1 and continuing without commentary."
    )
    filler_unit_ids = tokenizer.encode(" lorem", add_special_tokens=False)
    if not filler_unit_ids:
        filler_unit_ids = tokenizer.encode(" the", add_special_tokens=False)
    if not filler_unit_ids:
        raise RuntimeError("could not build filler token unit")

    def message_for_count(token_count: int) -> list[dict[str, str]]:
        repeated_ids = (filler_unit_ids * ((token_count + len(filler_unit_ids) - 1) // len(filler_unit_ids)))[:token_count]
        filler_text = tokenizer.decode(repeated_ids, clean_up_tokenization_spaces=False)
        return [{"role": "user", "content": prefix + filler_text + suffix}]

    def chat_len(token_count: int) -> int:
        tokenized = tokenizer.apply_chat_template(
            message_for_count(token_count),
            tokenize=True,
            add_generation_prompt=True,
        )
        return _token_count(tokenized)

    if target_tokens <= 0:
        messages = message_for_count(0)
        actual = _token_count(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )
        return messages, actual

    lo = 0
    hi = max(target_tokens, 1)
    while chat_len(hi) < target_tokens:
        hi *= 2

    best_count = hi
    best_delta = abs(chat_len(hi) - target_tokens)
    while lo <= hi:
        mid = (lo + hi) // 2
        total = chat_len(mid)
        delta = abs(total - target_tokens)
        if delta < best_delta:
            best_count = mid
            best_delta = delta
        if total < target_tokens:
            lo = mid + 1
        elif total > target_tokens:
            hi = mid - 1
        else:
            best_count = mid
            break

    for candidate in range(max(0, best_count - 16), best_count + 17):
        total = chat_len(candidate)
        delta = abs(total - target_tokens)
        if delta < best_delta:
            best_count = candidate
            best_delta = delta
            if delta == 0:
                break

    messages = message_for_count(best_count)
    actual = _token_count(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    )
    return messages, actual


async def _stream_one_request(
    client: httpx.AsyncClient,
    *,
    url: str,
    payload: dict[str, Any],
    started_at_s: float,
) -> StreamResult:
    result = StreamResult()
    request_start = time.monotonic()
    usage_tokens: int | None = None
    first_token_time: float | None = None
    chunk_count = 0
    try:
        async with client.stream(
            "POST",
            url,
            json=payload,
            timeout=httpx.Timeout(600.0, connect=30.0),
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                result.error = f"HTTP {resp.status_code}: {body.decode(errors='replace')[:200]}"
                result.total_time_s = time.monotonic() - request_start
                return result

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                usage = data.get("usage")
                if usage and "completion_tokens" in usage:
                    usage_tokens = int(usage["completion_tokens"])

                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = ""
                if delta.get("reasoning"):
                    text += delta["reasoning"]
                if delta.get("reasoning_content"):
                    text += delta["reasoning_content"]
                if delta.get("content"):
                    text += delta["content"]
                if text and first_token_time is None:
                    first_token_time = time.monotonic()
                if text:
                    chunk_count += 1
    except Exception as exc:  # pragma: no cover - runtime diagnostic path
        result.error = f"{type(exc).__name__}: {exc}"

    end_time = time.monotonic()
    result.total_time_s = end_time - request_start
    result.end_at_s = end_time - started_at_s
    result.total_tokens = usage_tokens if usage_tokens is not None else chunk_count
    if first_token_time is not None:
        result.ttft_s = first_token_time - request_start
        result.first_token_at_s = first_token_time - started_at_s
        result.decode_tokens = max(result.total_tokens - 1, 0)
        result.decode_time_s = max(end_time - first_token_time, 0.0)
    return result


async def _poll_metrics(
    base_url: str,
    stop_event: asyncio.Event,
    sample_store: dict[str, list[float]],
) -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
        while not stop_event.is_set():
            try:
                resp = await client.get(f"{base_url}/metrics")
                if resp.status_code == 200:
                    metrics = _parse_metrics(resp.text)
                    sample_store["gen_tps"].append(_extract_metric(metrics, "sglang:gen_throughput"))
                    sample_store["running"].append(_extract_metric(metrics, "sglang:num_running_reqs"))
                    sample_store["queue"].append(_extract_metric(metrics, "sglang:num_queue_reqs"))
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass


async def _warm_prefix_cache(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    warmup_max_tokens: int,
) -> None:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": warmup_max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream_options": {"include_usage": True},
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        await _stream_one_request(client, url=f"{base_url}/v1/chat/completions", payload=payload, started_at_s=time.monotonic())


async def run_regime(args: argparse.Namespace) -> RegimeSummary:
    model = args.model or await _wait_until_ready(args.base_url, args.ready_timeout)
    tokenizer = _make_tokenizer(args.tokenizer_path)
    messages, actual_tokens = _build_messages_for_context(tokenizer, args.context_tokens)

    # Warm compile/graph state before the measured cell.
    if not args.skip_warmup:
        await _warm_prefix_cache(
            args.base_url,
            model,
            messages,
            warmup_max_tokens=max(args.warmup_max_tokens, 1),
        )
        await _warm_prefix_cache(
            args.base_url,
            model,
            messages,
            warmup_max_tokens=1,
        )

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream_options": {"include_usage": True},
    }

    started_at_s = time.monotonic()
    stop_event = asyncio.Event()
    samples = {"gen_tps": [], "running": [], "queue": []}

    limits = httpx.Limits(
        max_connections=args.concurrency + 8,
        max_keepalive_connections=args.concurrency + 4,
    )
    async with httpx.AsyncClient(limits=limits) as client:
        poll_task = asyncio.create_task(_poll_metrics(args.base_url, stop_event, samples))
        try:
            tasks = [
                asyncio.create_task(
                    _stream_one_request(
                        client,
                        url=f"{args.base_url}/v1/chat/completions",
                        payload=payload,
                        started_at_s=started_at_s,
                    )
                )
                for _ in range(args.concurrency)
            ]
            results = await asyncio.gather(*tasks)
        finally:
            stop_event.set()
            await poll_task

    success = [result for result in results if result.error is None]
    failed = [result for result in results if result.error is not None]
    if not success:
        errors = [result.error for result in failed]
        raise RuntimeError(f"all requests failed: {errors}")

    ttfts = [result.ttft_s for result in success if result.ttft_s > 0]
    request_times = [result.total_time_s for result in success if result.total_time_s > 0]
    per_req_decode_tps = [
        result.decode_tokens / result.decode_time_s
        for result in success
        if result.decode_tokens > 0 and result.decode_time_s > 0
    ]

    total_completion_tokens = sum(result.total_tokens for result in success)
    total_decode_tokens = sum(result.decode_tokens for result in success)
    first_decode = min((result.first_token_at_s for result in success if result.first_token_at_s > 0), default=0.0)
    last_decode = max((result.end_at_s for result in success if result.end_at_s > 0), default=0.0)
    aggregate_client_decode_tps = 0.0
    if total_decode_tokens > 0 and last_decode > first_decode:
        aggregate_client_decode_tps = total_decode_tokens / (last_decode - first_decode)

    summary = RegimeSummary(
        label=args.label,
        base_url=args.base_url,
        model=model,
        concurrency=args.concurrency,
        context_tokens_target=args.context_tokens,
        context_tokens_actual=actual_tokens,
        max_tokens=args.max_tokens,
        successful_requests=len(success),
        failed_requests=len(failed),
        total_completion_tokens=total_completion_tokens,
        median_ttft_s=_median(ttfts),
        p95_ttft_s=_percentile(ttfts, 95.0),
        median_request_time_s=_median(request_times),
        median_request_decode_tps=_median(per_req_decode_tps),
        aggregate_client_decode_tps=aggregate_client_decode_tps,
        median_server_decode_tps=_median([value for value in samples["gen_tps"] if value > 0]),
        median_server_running_reqs=_median(samples["running"]),
        median_server_queue_reqs=_median(samples["queue"]),
    )

    print(json.dumps({"summary": asdict(summary), "requests": [asdict(r) for r in results]}, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": asdict(summary), "requests": [asdict(r) for r in results]}, f, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Server base URL, e.g. http://127.0.0.1:8010")
    parser.add_argument("--label", default="run", help="Label carried into the JSON summary")
    parser.add_argument("--model", default=None, help="Served model name. Defaults to /v1/models discovery.")
    parser.add_argument("--tokenizer-path", required=True, help="Local model/tokenizer path for prompt construction")
    parser.add_argument("--context-tokens", type=int, default=128, help="Target prompt token count after chat templating")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of simultaneous decode requests")
    parser.add_argument("--max-tokens", type=int, default=128, help="Generated tokens per request")
    parser.add_argument("--warmup-max-tokens", type=int, default=64, help="Warmup decode length before the measured cell")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip the compile/graph warmup requests")
    parser.add_argument("--ready-timeout", type=float, default=300.0, help="Seconds to wait for /v1/models")
    parser.add_argument("--output-json", default=None, help="Optional path to save the raw JSON result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_regime(args))


if __name__ == "__main__":
    main()
