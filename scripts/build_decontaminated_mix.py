#!/usr/bin/env python3
"""Build the decontaminated 68k multi-domain on-policy training & validation corpus for KvN-DSpark v2.

Implements Phase B of ``kvn_rff_rnn_mtp_train_phase_2.md``:
1. Pulls strictly from **training** splits of three complementary domains:
   - ``HuggingFaceH4/ultrachat_200k`` (``train_sft``): General dialogue & multi-turn instruction following
   - ``nvidia/OpenMathInstruct-2`` (``train``): Long-horizon mathematical & CoT reasoning
   - ``ise-uiuc/Magicoder-OSS-Instruct-75K`` (``train``): Code generation & algorithmic structure
2. Enforces **strict SHA-256 + normalized 13-gram decontamination** against downstream evaluation
   benchmarks (MT-Bench, GSM8K test, MATH-500, HumanEval, MBPP, LiveCodeBench, plus any custom JSONL).
3. Assigns stratified sampling temperatures matching the Phase 2 protocol:
   - UltraChat: 50% T=0.0 / 50% T=0.7
   - OpenMathInstruct-2: 50% T=0.0 / 50% T=0.6
   - Magicoder: 60% T=0.0 / 40% T=0.6
4. Optionally regenerates every assistant turn **on-policy** against a live OpenAI-compatible
   ``vLLM`` server (``--vllm-endpoint``) and re-runs 13-gram decontamination on generated completions
   before writing ``speculators prepare-data``-ready JSONL files.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import unicodedata
import urllib.request

logger = logging.getLogger("build_decontaminated_mix")

_WORD_RE = re.compile(r"[a-z0-9]+")

# Hugging Face evaluation benchmark specifications used solely to build the exclusion filter.
BENCHMARK_HF_SPECS: dict[str, tuple[str, str | None, str, tuple[str, ...]]] = {
    "gsm8k": ("openai/gsm8k", "main", "test", ("question", "answer")),
    "math500": ("HuggingFaceH4/MATH-500", None, "test", ("problem", "solution")),
    "humaneval": ("openai/openai_humaneval", None, "test", ("prompt", "canonical_solution")),
    "mbpp": ("google-research-datasets/mbpp", "sanitized", "test", ("prompt", "code")),
    "mt_bench": ("HuggingFaceH4/mt_bench_prompts", None, "train", ("prompt",)),
    "livecodebench": (
        "livecodebench/code_generation_lite",
        None,
        "test",
        ("question_content",),
    ),
}

DOMAIN_HF_SPECS: dict[str, tuple[str, str | None, str]] = {
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft"),
    "openmath": ("nvidia/OpenMathInstruct-2", None, "train"),
    "magicoder": ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train"),
}


def normalize_for_decontamination(text: str) -> list[str]:
    """Normalize text to lowercase alphanumeric tokens (NFKC) for SHA-256 & n-gram matching."""
    norm = unicodedata.normalize("NFKC", text).lower()
    return _WORD_RE.findall(norm)


def _hash_tokens(tokens: list[str] | tuple[str, ...]) -> bytes:
    joined = " ".join(tokens).encode("utf-8")
    return hashlib.sha256(joined).digest()[:16]


@dataclass
class BenchmarkContaminationIndex:
    """Exact normalized-prompt SHA-256 + rolling 13-gram index over evaluation benchmarks."""

    ngram_size: int = 13
    exact_hashes: dict[bytes, str] = field(default_factory=dict)
    ngram_hashes: dict[bytes, str] = field(default_factory=dict)

    def add_benchmark_text(self, benchmark_name: str, text: str | list[str]) -> None:
        if isinstance(text, list):
            for item in text:
                if item:
                    self.add_benchmark_text(benchmark_name, str(item))
            return
        tokens = normalize_for_decontamination(text)
        if not tokens:
            return
        self.exact_hashes.setdefault(_hash_tokens(tokens), benchmark_name)
        n = self.ngram_size
        if len(tokens) >= n:
            for idx in range(len(tokens) - n + 1):
                h = _hash_tokens(tokens[idx : idx + n])
                self.ngram_hashes.setdefault(h, benchmark_name)

    def is_contaminated(self, conversations: list[dict[str, str]]) -> tuple[bool, str]:
        """Return ``(True, reason)`` if any turn collides by SHA-256 or n-gram overlap."""
        n = self.ngram_size
        for turn in conversations:
            content = str(turn.get("content", ""))
            tokens = normalize_for_decontamination(content)
            if not tokens:
                continue
            full_hash = _hash_tokens(tokens)
            if full_hash in self.exact_hashes:
                return True, f"sha256_exact:{self.exact_hashes[full_hash]}"
            if len(tokens) >= n:
                for idx in range(len(tokens) - n + 1):
                    ngram_h = _hash_tokens(tokens[idx : idx + n])
                    if ngram_h in self.ngram_hashes:
                        return True, f"{n}gram:{self.ngram_hashes[ngram_h]}"
        return False, ""


def normalize_domain_row(domain: str, row: dict) -> list[dict[str, str]] | None:
    """Convert raw domain dataset record into canonical ``[{'role': ..., 'content': ...}]``."""
    if domain == "ultrachat":
        raw_msgs = row.get("messages") or row.get("conversations") or []
        conv: list[dict[str, str]] = []
        for turn in raw_msgs:
            role = str(turn.get("role") or turn.get("from") or "").strip().lower()
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            content = str(turn.get("content") or turn.get("value") or "").strip()
            if role in {"system", "user", "assistant"} and content:
                conv.append({"role": role, "content": content})
        has_user = any(t["role"] == "user" for t in conv)
        has_assistant = any(t["role"] == "assistant" for t in conv)
        return conv if (has_user and has_assistant) else None

    if domain == "openmath":
        problem = str(row.get("problem") or row.get("question") or "").strip()
        solution = str(
            row.get("generated_solution") or row.get("solution") or row.get("answer") or ""
        ).strip()
        if not problem or not solution:
            return None
        return [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]

    if domain == "magicoder":
        problem = str(row.get("problem") or row.get("instruction") or "").strip()
        solution = str(row.get("solution") or row.get("response") or "").strip()
        if not problem or not solution:
            return None
        return [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]

    raise ValueError(f"Unsupported domain: {domain!r}")


def assign_sampling_temperature(domain: str, sample_index: int) -> float:
    """Deterministic stratified sampling temperature per Phase B table."""
    slot = sample_index % 10
    if domain == "ultrachat":
        return 0.0 if slot < 5 else 0.7
    if domain == "openmath":
        return 0.0 if slot < 5 else 0.6
    if domain == "magicoder":
        return 0.0 if slot < 6 else 0.6
    raise ValueError(f"Unsupported domain: {domain!r}")


def load_benchmark_index(
    benchmarks: list[str],
    extra_jsonl_paths: list[Path] | None = None,
    ngram_size: int = 13,
) -> BenchmarkContaminationIndex:
    """Populate the decontamination index from Hugging Face test splits + optional local JSONLs."""
    from datasets import load_dataset

    index = BenchmarkContaminationIndex(ngram_size=ngram_size)
    for bench in benchmarks:
        if bench not in BENCHMARK_HF_SPECS:
            logger.warning("Unknown benchmark %r skipped.", bench)
            continue
        hf_path, subset, split, fields = BENCHMARK_HF_SPECS[bench]
        logger.info("Indexing benchmark %s (%s [%s])...", bench, hf_path, split)
        ds = (
            load_dataset(hf_path, subset, split=split)
            if subset is not None
            else load_dataset(hf_path, split=split)
        )
        for row in ds:
            for col in fields:
                val = row.get(col)
                if val:
                    index.add_benchmark_text(bench, val)

    for jsonl_path in extra_jsonl_paths or []:
        with Path(jsonl_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for val in record.values():
                    if isinstance(val, str):
                        index.add_benchmark_text(jsonl_path.stem, val)

    logger.info(
        "Decontamination index ready: %d exact hashes, %d %d-grams.",
        len(index.exact_hashes),
        len(index.ngram_hashes),
        ngram_size,
    )
    return index


def _chat_completion_sync(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: float = 300.0,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if temperature == 0.0:
        payload["top_p"] = 1.0
    req = urllib.request.Request(  # noqa: S310
        f"{endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        body = json.loads(resp.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"]).strip()


async def regenerate_conversation_on_policy(
    sem: asyncio.Semaphore,
    endpoint: str,
    model: str,
    record: dict,
    max_tokens: int,
) -> dict | None:
    """Regenerate every assistant turn sequentially using on-policy verifier completions."""
    temperature = float(record["temperature"])
    context: list[dict[str, str]] = []
    async with sem:
        for turn in record["conversations"]:
            if turn["role"] != "assistant":
                context.append(turn)
                continue
            try:
                reply = await asyncio.to_thread(
                    _chat_completion_sync,
                    endpoint,
                    model,
                    list(context),
                    temperature,
                    max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("On-policy generation failed (%s); dropping sample.", exc)
                return None
            if not reply:
                return None
            context.append({"role": "assistant", "content": reply})
    return {**record, "conversations": context}


async def regenerate_batch_async(
    records: list[dict],
    endpoint: str,
    model: str,
    concurrency: int = 64,
    max_tokens: int = 4096,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        regenerate_conversation_on_policy(sem, endpoint, model, rec, max_tokens)
        for rec in records
    ]
    completed = await asyncio.gather(*tasks)
    return [r for r in completed if r is not None]


def collect_decontaminated_domain_samples(
    domain: str,
    target_count: int,
    index: BenchmarkContaminationIndex,
    seed: int = 42,
) -> tuple[list[dict], int]:
    """Stream a training split from HF Hub and retain ``target_count`` decontaminated conversations."""
    from datasets import load_dataset

    if target_count <= 0:
        return [], 0
    hf_path, subset, split = DOMAIN_HF_SPECS[domain]
    logger.info("Streaming %s (%s [%s]) for %d clean samples...", domain, hf_path, split, target_count)
    ds = (
        load_dataset(hf_path, subset, split=split, streaming=True)
        if subset is not None
        else load_dataset(hf_path, split=split, streaming=True)
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    accepted: list[dict] = []
    rejected_leaks = 0
    for row in ds:
        conv = normalize_domain_row(domain, row)
        if conv is None:
            continue
        contaminated, _reason = index.is_contaminated(conv)
        if contaminated:
            rejected_leaks += 1
            continue
        temp = assign_sampling_temperature(domain, len(accepted))
        accepted.append(
            {
                "conversations": conv,
                "domain": domain,
                "temperature": temp,
            }
        )
        if len(accepted) >= target_count:
            break

    logger.info(
        "Domain %s: retained %d clean samples (%d contaminated candidates rejected).",
        domain,
        len(accepted),
        rejected_leaks,
    )
    return accepted, rejected_leaks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the decontaminated 68k multi-domain corpus for KvN-DSpark Phase 2."
    )
    parser.add_argument("--verifier-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--ultrachat-samples", type=int, default=31500)
    parser.add_argument("--openmath-samples", type=int, default=21000)
    parser.add_argument("--magicoder-samples", type=int, default=15500)
    parser.add_argument(
        "--exclude-benchmarks",
        nargs="+",
        default=["mt_bench", "gsm8k", "math500", "humaneval", "mbpp", "livecodebench"],
    )
    parser.add_argument("--extra-benchmark-jsonl", nargs="*", type=Path, default=[])
    parser.add_argument("--ngram-size", type=int, default=13)
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default=None,
        help="Optional live vLLM OpenAI endpoint (e.g. http://localhost:8000/v1) for on-policy regeneration.",
    )
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--val-output-jsonl",
        type=Path,
        default=None,
        help="Optional separate path for the stratified held-out validation split.",
    )
    parser.add_argument("--val-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    args = parse_args(argv)
    rng = random.Random(args.seed)

    index = load_benchmark_index(
        benchmarks=list(args.exclude_benchmarks),
        extra_jsonl_paths=list(args.extra_benchmark_jsonl),
        ngram_size=args.ngram_size,
    )

    domain_targets = {
        "ultrachat": args.ultrachat_samples,
        "openmath": args.openmath_samples,
        "magicoder": args.magicoder_samples,
    }

    all_records: list[dict] = []
    total_rejected = 0
    for domain, count in domain_targets.items():
        recs, n_rej = collect_decontaminated_domain_samples(
            domain=domain, target_count=count, index=index, seed=args.seed
        )
        if args.vllm_endpoint:
            logger.info(
                "Regenerating %d %s conversations on-policy via %s...",
                len(recs),
                domain,
                args.vllm_endpoint,
            )
            recs = asyncio.run(
                regenerate_batch_async(
                    records=recs,
                    endpoint=args.vllm_endpoint,
                    model=args.verifier_model,
                    concurrency=args.concurrency,
                    max_tokens=args.max_completion_tokens,
                )
            )
            # Re-verify generated assistant turns against 13-gram benchmark overlap
            post_clean: list[dict] = []
            for rec in recs:
                contaminated, _ = index.is_contaminated(rec["conversations"])
                if contaminated:
                    total_rejected += 1
                else:
                    post_clean.append(rec)
            recs = post_clean

        all_records.extend(recs)
        total_rejected += n_rej

    rng.shuffle(all_records)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.val_output_jsonl is not None and 0 < args.val_samples < len(all_records):
        val_records = all_records[: args.val_samples]
        train_records = all_records[args.val_samples :]
        args.val_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.val_output_jsonl.open("w", encoding="utf-8", newline="\n") as fh:
            for rec in val_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Wrote %d held-out validation conversations to %s", len(val_records), args.val_output_jsonl)
    else:
        train_records = all_records

    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in train_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest_path = args.output_jsonl.with_suffix(".manifest.json")
    manifest = {
        "verifier_model": args.verifier_model,
        "seed": args.seed,
        "ngram_size": args.ngram_size,
        "excluded_benchmarks": args.exclude_benchmarks,
        "on_policy_regenerated": bool(args.vllm_endpoint),
        "total_written_train": len(train_records),
        "total_written_val": (
            len(all_records) - len(train_records) if args.val_output_jsonl else 0
        ),
        "contaminated_candidates_rejected": total_rejected,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Completed mix build: %s (manifest: %s)", args.output_jsonl, manifest_path)


if __name__ == "__main__":
    main()
