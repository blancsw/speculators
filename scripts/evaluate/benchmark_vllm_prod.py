#!/usr/bin/env python3
"""Production ``vllm serve`` Speculative Decoding & Zero-Replay APC Benchmark Client.

Benchmarks a live ``vLLM V1`` OpenAI-compatible server (or launches ``vllm serve`` as a
managed subprocess) across concurrency sweeps ``B in {1, 4, 8, 16, 32, 64, 128}``:

- Streams ``/v1/chat/completions`` SSE token chunks via ``httpx.AsyncClient`` to measure
  per-request **TTFT** (Time-To-First-Token), **ITL** (Inter-Token Latency), **TPOT**
  (Time-Per-Output-Token), aggregate output throughput (``tok/s``), and end-to-end
  speedup ``S_e2e(B)`` against a non-speculative baseline endpoint/run;
- Scrapes ``/metrics`` Prometheus counters (``vllm:spec_decode_num_accepted_tokens`` /
  ``vllm:spec_decode_num_drafts``) before & after each concurrency tier to report exact
  live empirical acceptance length ``tau = 1 + accepted / drafts``;
- Evaluates **Axis 3 Zero-Replay APC** multi-turn / shared-system-prompt prefix cache
  hit speedup (cold-cache vs warm-cache TTFT & throughput);
- Optionally delegates load generation to ``aiperf profile`` via ``--use-aiperf``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx

DEFAULT_CONCURRENCIES = (1, 4, 8, 16, 32, 64, 128)

DEFAULT_SHARED_SYSTEM_PREFIX = (
    "You are a Principal Systems Architect and Mathematical Physicist specializing "
    "in Koopman-von Neumann Bosonic state-space models, symplectic integrators, "
    "PagedAttention memory hierarchies, and lossless speculative decoding in vLLM "
    "V1. Provide rigorous, concise derivations with explicit complexity and "
    "byte-budget bounds.\n"
) * 6

BENCHMARK_USER_PROMPTS = (
    "Derive the roofline ridge point of an H200 SXM GPU (989 BF16 TFLOPS, 4.8 TB/s "
    "HBM3e) and explain why speculative verification of K=7 draft tokens is "
    "latency-free at B<=24.",
    "Prove that SpecInfer sequential residual rejection sampling "
    "p_res <- max(0, p - q) preserves the exact target distribution p(y|x) at "
    "temperature T = 0.6.",
    "Compare 128-token SWA Bounded Replay prefix caching against 0-FLOP Markovian "
    "Bosonic recurrent state checkpointing at 16-token PagedAttention block "
    "boundaries.",
    "Write a self-contained PyTorch function implementing a parallel Kogge-Stone "
    "associative prefix scan for 2x2 damped harmonic rotations in float32.",
)


@dataclass
class RequestLatencyRecord:
    """Per-request streaming latency & token accounting."""

    ttft_ms: float
    e2e_ms: float
    tpot_ms: float
    itl_ms_list: list[float]
    output_tokens: int


@dataclass
class ConcurrencyTierSummary:
    """Aggregated production metrics at concurrency ``B``."""

    concurrency: int
    num_requests: int
    wall_time_s: float
    total_output_tokens: int
    output_throughput_tok_s: float
    mean_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    mean_tpot_ms: float
    mean_itl_ms: float
    p95_itl_ms: float
    mean_accept_length_tau: float | None = None
    draft_acceptance_rate: float | None = None
    speedup_vs_baseline: float | None = None


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * max(0.0, min(1.0, q))
    lo = int(idx)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def parse_prometheus_spec_decode_counters(metrics_text: str) -> dict[str, float]:
    """Extract cumulative vLLM speculative decoding counters from ``/metrics`` text."""
    totals: dict[str, float] = {
        "num_drafts": 0.0,
        "num_draft_tokens": 0.0,
        "num_accepted_tokens": 0.0,
    }
    patterns = {
        "num_drafts": re.compile(
            r"^vllm:spec_decode_num_drafts(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)"
        ),
        "num_draft_tokens": re.compile(
            r"^vllm:spec_decode_num_draft_tokens(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)"
        ),
        "num_accepted_tokens": re.compile(
            r"^vllm:spec_decode_num_accepted_tokens(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)"
        ),
    }
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for key, pat in patterns.items():
            m = pat.match(line)
            if m:
                totals[key] += float(m.group(1))
    return totals


def compute_delta_spec_metrics(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
) -> tuple[float | None, float | None]:
    """Compute ``(mean_accept_length_tau, draft_acceptance_rate)`` from deltas."""
    if before is None or after is None:
        return None, None
    d_drafts = after["num_drafts"] - before["num_drafts"]
    d_draft_toks = after["num_draft_tokens"] - before["num_draft_tokens"]
    d_accepted = after["num_accepted_tokens"] - before["num_accepted_tokens"]
    if d_drafts <= 0:
        return None, None
    tau = 1.0 + (d_accepted / d_drafts)
    acc_rate = (d_accepted / d_draft_toks) if d_draft_toks > 0 else None
    return round(tau, 4), (round(acc_rate, 4) if acc_rate is not None else None)


async def fetch_prometheus_counters(
    client: httpx.AsyncClient,
    base_url: str,
) -> dict[str, float] | None:
    """Query ``{base_url}/metrics`` if exposed by the ``vllm serve`` instance."""
    try:
        resp = await client.get(f"{base_url.rstrip('/')}/metrics", timeout=10.0)
        if resp.status_code == HTTPStatus.OK:
            return parse_prometheus_spec_decode_counters(resp.text)
    except httpx.HTTPError:
        return None
    return None


async def stream_single_chat_request(
    client: httpx.AsyncClient,
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    api_key: str = "EMPTY",
) -> RequestLatencyRecord:
    """Send one streaming ``/v1/chat/completions`` request and record TTFT/ITL/TPOT."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first: float | None = None
    t_prev: float | None = None
    itl_ms: list[float] = []
    chunk_tokens = 0
    usage_completion_tokens: int | None = None

    async with client.stream(
        "POST", url, json=payload, headers=headers, timeout=300.0
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                packet = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            usage = packet.get("usage")
            if isinstance(usage, dict) and "completion_tokens" in usage:
                usage_completion_tokens = int(usage["completion_tokens"])

            choices = packet.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or delta.get("reasoning_content") or ""
            if not content:
                continue

            now = time.perf_counter()
            if t_first is None:
                t_first = now
            elif t_prev is not None:
                itl_ms.append((now - t_prev) * 1000.0)
            t_prev = now
            chunk_tokens += 1

    t_end = time.perf_counter()
    e2e_ms = (t_end - t_start) * 1000.0
    ttft_ms = ((t_first - t_start) * 1000.0) if t_first is not None else e2e_ms
    out_tokens = max(1, usage_completion_tokens or chunk_tokens)
    decode_window_ms = max(0.0, e2e_ms - ttft_ms)
    tpot_ms = decode_window_ms / max(1, out_tokens - 1) if out_tokens > 1 else 0.0

    return RequestLatencyRecord(
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
        tpot_ms=tpot_ms,
        itl_ms_list=itl_ms,
        output_tokens=out_tokens,
    )


def summarize_tier_records(
    concurrency: int,
    wall_time_s: float,
    records: list[RequestLatencyRecord],
    tau: float | None = None,
    acc_rate: float | None = None,
    baseline_tok_s: float | None = None,
) -> ConcurrencyTierSummary:
    """Aggregate streaming request latency records into a ``ConcurrencyTierSummary``."""
    total_toks = sum(r.output_tokens for r in records)
    tok_s = total_toks / max(wall_time_s, 1e-6)
    ttfts = sorted(r.ttft_ms for r in records)
    tpots = [r.tpot_ms for r in records if r.output_tokens > 1]
    all_itls = sorted(x for r in records for x in r.itl_ms_list)

    mean_ttft = sum(ttfts) / max(1, len(ttfts))
    mean_tpot = sum(tpots) / max(1, len(tpots)) if tpots else 0.0
    mean_itl = sum(all_itls) / max(1, len(all_itls)) if all_itls else mean_tpot
    speedup = (
        round(tok_s / baseline_tok_s, 4)
        if (baseline_tok_s is not None and baseline_tok_s > 0)
        else None
    )

    return ConcurrencyTierSummary(
        concurrency=concurrency,
        num_requests=len(records),
        wall_time_s=round(wall_time_s, 4),
        total_output_tokens=total_toks,
        output_throughput_tok_s=round(tok_s, 2),
        mean_ttft_ms=round(mean_ttft, 2),
        p50_ttft_ms=round(_percentile(ttfts, 0.50), 2),
        p95_ttft_ms=round(_percentile(ttfts, 0.95), 2),
        mean_tpot_ms=round(mean_tpot, 2),
        mean_itl_ms=round(mean_itl, 2),
        p95_itl_ms=round(_percentile(all_itls, 0.95), 2) if all_itls else 0.0,
        mean_accept_length_tau=tau,
        draft_acceptance_rate=acc_rate,
        speedup_vs_baseline=speedup,
    )


async def run_concurrency_tier(
    base_url: str,
    model_name: str,
    concurrency: int,
    requests_per_worker: int = 2,
    max_tokens: int = 128,
    temperature: float = 0.0,
    shared_prefix: str = "",
) -> ConcurrencyTierSummary:
    """Execute ``concurrency * requests_per_worker`` bounded-concurrency requests."""
    total_reqs = max(1, int(concurrency) * int(requests_per_worker))
    sem = asyncio.Semaphore(int(concurrency))
    limits = httpx.Limits(
        max_connections=max(32, int(concurrency) * 2),
        max_keepalive_connections=max(16, int(concurrency)),
    )

    async with httpx.AsyncClient(limits=limits) as client:
        counters_before = await fetch_prometheus_counters(client, base_url)

        async def _worker(req_idx: int) -> RequestLatencyRecord:
            prompt = BENCHMARK_USER_PROMPTS[req_idx % len(BENCHMARK_USER_PROMPTS)]
            messages: list[dict[str, str]] = []
            if shared_prefix:
                messages.append({"role": "system", "content": shared_prefix})
            messages.append(
                {"role": "user", "content": f"[Request #{req_idx}] {prompt}"}
            )
            async with sem:
                return await stream_single_chat_request(
                    client=client,
                    base_url=base_url,
                    model_name=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

        t0 = time.perf_counter()
        records = await asyncio.gather(*(_worker(i) for i in range(total_reqs)))
        wall_s = time.perf_counter() - t0
        counters_after = await fetch_prometheus_counters(client, base_url)

    tau, acc_rate = compute_delta_spec_metrics(counters_before, counters_after)
    return summarize_tier_records(
        concurrency=concurrency,
        wall_time_s=wall_s,
        records=list(records),
        tau=tau,
        acc_rate=acc_rate,
    )


async def benchmark_apc_prefix_hit_speedup(
    base_url: str,
    model_name: str,
    concurrency: int = 4,
    max_tokens: int = 64,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Compare cold vs warm prefix TTFT & throughput for Axis 3 Zero-Replay APC."""
    unique_nonce = f"[Session-{time.time_ns()}]\n"
    long_shared_prefix = unique_nonce + (DEFAULT_SHARED_SYSTEM_PREFIX * 4)

    cold_summary = await run_concurrency_tier(
        base_url=base_url,
        model_name=model_name,
        concurrency=1,
        requests_per_worker=1,
        max_tokens=max_tokens,
        temperature=temperature,
        shared_prefix=long_shared_prefix,
    )
    warm_summary = await run_concurrency_tier(
        base_url=base_url,
        model_name=model_name,
        concurrency=concurrency,
        requests_per_worker=2,
        max_tokens=max_tokens,
        temperature=temperature,
        shared_prefix=long_shared_prefix,
    )
    ttft_speedup = cold_summary.mean_ttft_ms / max(warm_summary.mean_ttft_ms, 1e-6)
    return {
        "cold_prefix_ttft_ms": cold_summary.mean_ttft_ms,
        "warm_prefix_ttft_ms": warm_summary.mean_ttft_ms,
        "apc_ttft_speedup": round(ttft_speedup, 4),
        "cold_summary": asdict(cold_summary),
        "warm_summary": asdict(warm_summary),
    }


def wait_for_vllm_ready(base_url: str, timeout_s: float = 300.0) -> None:
    """Poll ``{base_url}/health`` until HTTP 200 or timeout."""
    deadline = time.monotonic() + timeout_s
    health_url = f"{base_url.rstrip('/')}/health"
    while time.monotonic() < deadline:
        try:
            r = httpx.get(health_url, timeout=3.0)
            if r.status_code == HTTPStatus.OK:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    raise TimeoutError(
        f"vLLM server at {base_url} did not become healthy within {timeout_s}s."
    )


def launch_managed_vllm_server(
    target_model: str,
    port: int,
    speculative_config_json: str | None = None,
    enable_prefix_caching: bool = True,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 8192,
) -> subprocess.Popen[str]:
    """Launch ``vllm serve`` in a background subprocess."""
    vllm_bin = shutil.which("vllm")
    if vllm_bin is None:
        raise FileNotFoundError("`vllm` CLI binary not found in PATH.")

    cmd = [
        vllm_bin,
        "serve",
        target_model,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
    ]
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")

    env = os.environ.copy()
    if speculative_config_json:
        try:
            parsed = json.loads(speculative_config_json)
            ckpt = parsed.get("model")
            if isinstance(ckpt, str) and Path(ckpt).exists():
                env.setdefault("KVN_DSPARK_CHECKPOINT_PATH", str(Path(ckpt).resolve()))
        except json.JSONDecodeError:
            pass
        cmd.extend(["--speculative-config", speculative_config_json])

    return subprocess.Popen(cmd, env=env, text=True)  # noqa: S603


def run_aiperf_profile(
    base_url: str,
    model_name: str,
    concurrency: int,
    output_dir: Path,
) -> int:
    """Invoke NVIDIA ``aiperf profile`` against ``base_url`` when installed."""
    aiperf_bin = shutil.which("aiperf")
    if aiperf_bin is None:
        raise FileNotFoundError("`aiperf` CLI binary is not installed in PATH.")
    tier_dir = output_dir / f"aiperf_c{concurrency}"
    tier_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        aiperf_bin,
        "profile",
        "--model",
        model_name,
        "--url",
        base_url,
        "--endpoint-type",
        "chat",
        "--streaming",
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(max(8, concurrency * 2)),
        "--artifact-dir",
        str(tier_dir),
    ]
    completed = subprocess.run(cmd, check=False)  # noqa: S603
    return completed.returncode


def _extract_sweep_tps(entries: list[Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in entries:
        if isinstance(item, dict) and "concurrency" in item:
            tps = float(item.get("output_throughput_tok_s", 0.0) or 0.0)
            if tps > 0.0:
                out[int(item["concurrency"])] = tps
    return out


def _extract_baseline_map_tps(base_map: dict[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for c_raw, item in base_map.items():
        if isinstance(item, dict):
            tps = float(item.get("output_throughput_tok_s", 0.0) or 0.0)
            if tps > 0.0:
                out[int(item.get("concurrency", c_raw))] = tps
    return out


def load_baseline_throughput_map(baseline_json: Path | None) -> dict[int, float]:
    """Load per-concurrency baseline ``output_throughput_tok_s`` from saved JSON."""
    if baseline_json is None or not Path(baseline_json).is_file():
        return {}
    try:
        payload = json.loads(Path(baseline_json).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    sweep = payload.get("concurrency_sweep")
    tps_by_c = _extract_sweep_tps(sweep) if isinstance(sweep, list) else {}
    if not tps_by_c and isinstance(payload.get("baseline_sweep"), dict):
        tps_by_c = _extract_baseline_map_tps(payload["baseline_sweep"])
    return tps_by_c


async def run_full_benchmark_suite(args: argparse.Namespace) -> dict[str, Any]:
    """Run warmup, concurrency sweep, baseline speedup pairing, and APC benchmark."""
    concurrencies = [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]

    # Warmup requests (>= 5 iterations)
    await run_concurrency_tier(
        base_url=args.base_url,
        model_name=args.model_name,
        concurrency=1,
        requests_per_worker=max(5, args.warmup_requests),
        max_tokens=min(32, args.max_tokens),
        temperature=args.temperature,
    )

    baseline_json_path: Path | None = getattr(args, "baseline_json", None)
    baseline_tps_by_c: dict[int, float] = load_baseline_throughput_map(
        baseline_json_path
    )
    baseline_by_c: dict[int, ConcurrencyTierSummary] = {}
    if args.baseline_base_url:
        await run_concurrency_tier(
            base_url=args.baseline_base_url,
            model_name=args.model_name,
            concurrency=1,
            requests_per_worker=max(5, args.warmup_requests),
            max_tokens=min(32, args.max_tokens),
            temperature=args.temperature,
        )
        for c_val in concurrencies:
            tier_base = await run_concurrency_tier(
                base_url=args.baseline_base_url,
                model_name=args.model_name,
                concurrency=c_val,
                requests_per_worker=args.requests_per_worker,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            baseline_by_c[c_val] = tier_base
            if tier_base.output_throughput_tok_s > 0.0:
                baseline_tps_by_c[c_val] = tier_base.output_throughput_tok_s

    spec_summaries: list[dict[str, Any]] = []
    for c_val in concurrencies:
        tier = await run_concurrency_tier(
            base_url=args.base_url,
            model_name=args.model_name,
            concurrency=c_val,
            requests_per_worker=args.requests_per_worker,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        base_tps = baseline_tps_by_c.get(c_val)
        if base_tps is not None and base_tps > 0:
            tier.speedup_vs_baseline = round(tier.output_throughput_tok_s / base_tps, 4)
        spec_summaries.append(asdict(tier))

        if args.use_aiperf and args.output_json:
            run_aiperf_profile(
                base_url=args.base_url,
                model_name=args.model_name,
                concurrency=c_val,
                output_dir=Path(args.output_json).parent,
            )

    apc_report = None
    if not args.skip_apc_benchmark:
        apc_report = await benchmark_apc_prefix_hit_speedup(
            base_url=args.base_url,
            model_name=args.model_name,
            concurrency=min(4, max(concurrencies)),
            max_tokens=min(64, args.max_tokens),
            temperature=args.temperature,
        )

    return {
        "model_name": args.model_name,
        "base_url": args.base_url,
        "baseline_base_url": args.baseline_base_url,
        "baseline_json": str(baseline_json_path) if baseline_json_path else None,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "concurrency_sweep": spec_summaries,
        "baseline_sweep": {str(k): asdict(v) for k, v in baseline_by_c.items()},
        "apc_prefix_benchmark": apc_report,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production vllm serve Speculative Decoding & APC Benchmark."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Target speculative vLLM OpenAI server URL.",
    )
    parser.add_argument(
        "--baseline-base-url",
        type=str,
        default=None,
        help="Optional non-speculative baseline vLLM URL for exact S_e2e(B) ratios.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional saved baseline JSON for sequential single-GPU S_e2e(B).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Served model identifier passed to /v1/chat/completions.",
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        default=",".join(str(c) for c in DEFAULT_CONCURRENCIES),
        help="Comma-separated concurrency sweep.",
    )
    parser.add_argument("--requests-per-worker", type=int, default=2)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-apc-benchmark", action="store_true")
    parser.add_argument("--use-aiperf", action="store_true")
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--speculative-config", type=str, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    proc: subprocess.Popen[str] | None = None
    try:
        if args.launch_server:
            args.base_url = f"http://127.0.0.1:{args.port}"
            proc = launch_managed_vllm_server(
                target_model=args.model_name,
                port=args.port,
                speculative_config_json=args.speculative_config,
            )
            wait_for_vllm_ready(args.base_url)

        results = asyncio.run(run_full_benchmark_suite(args))
        formatted = json.dumps(results, indent=2)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(formatted, encoding="utf-8")
        print(formatted)
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
