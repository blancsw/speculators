#!/usr/bin/env python3
"""Build the decontaminated 100k multi-domain on-policy training & validation corpus for KvN-DSpark v2.

Implements Phase B of ``kvn_rff_rnn_mtp_train_phase_2.md``:
1. Pulls strictly from **training** splits of three complementary domains (100,000 total):
   - ``HuggingFaceH4/ultrachat_200k`` (``train_sft``, 45k / 207.9k): General dialogue & multi-turn instruction following
   - ``nvidia/OpenMathInstruct-2`` (``train``, 32k / 14M): Long-horizon mathematical & CoT reasoning
   - ``ise-uiuc/Magicoder-OSS-Instruct-75K`` (``train``, 23k / 75.2k): Code generation & algorithmic structure
2. Enforces **strict SHA-256 + normalized 13-gram decontamination** against downstream evaluation
   benchmarks (MT-Bench, GSM8K test, MATH-500, HumanEval, MBPP, LiveCodeBench, plus any custom JSONL),
   with automatic JSONL fallback for Hugging Face repos whose legacy Python loading scripts are
   deprecated under ``datasets >= 3.0`` (e.g. ``livecodebench/code_generation_lite``).
3. Assigns stratified sampling temperatures matching the Phase 2 protocol:
   - UltraChat: 50% T=0.0 / 50% T=0.7
   - OpenMathInstruct-2: 50% T=0.0 / 50% T=0.6
   - Magicoder: 60% T=0.0 / 40% T=0.6
4. Optionally regenerates assistant turns **on-policy** against a live OpenAI-compatible ``vLLM``
   server (``--vllm-endpoint``) using pooled ``aiohttp`` + ``"return_token_ids": True`` to emit
   **token-exact pretokenized ``(input_ids, loss_mask)`` boundary rows**, re-verifies generated
   completions against 13-gram benchmark overlap, streams replacements until exact per-domain
   row counts (45k / 32k / 23k = 100k) are reached, and supports crash-safe ``--resume``.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import random
import re
from typing import Any
import unicodedata

import aiohttp

logger = logging.getLogger("build_decontaminated_mix")

_WORD_RE = re.compile(r"[a-z0-9]+")

# Hugging Face evaluation benchmark specifications used solely to build the exclusion filter.
BENCHMARK_HF_SPECS: dict[str, tuple[str, str | None, str, tuple[str, ...]]] = {
    "gsm8k": ("openai/gsm8k", "main", "test", ("question", "answer")),
    "math500": ("HuggingFaceH4/MATH-500", None, "test", ("problem", "solution")),
    "humaneval": (
        "openai/openai_humaneval",
        None,
        "test",
        ("prompt", "canonical_solution"),
    ),
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
            row.get("generated_solution")
            or row.get("solution")
            or row.get("answer")
            or ""
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


def _iter_hf_repo_jsonl_rows(hf_path: str) -> Iterator[dict[str, Any]]:
    """Script-less fallback: download raw JSONL files directly via huggingface_hub."""
    import huggingface_hub

    repo_files = huggingface_hub.list_repo_files(hf_path, repo_type="dataset")
    jsonl_files = sorted(f for f in repo_files if f.endswith(".jsonl"))
    if not jsonl_files:
        raise RuntimeError(f"No .jsonl files found in dataset repo {hf_path!r}")
    for filename in jsonl_files:
        local_path = huggingface_hub.hf_hub_download(
            repo_id=hf_path, filename=filename, repo_type="dataset"
        )
        with Path(local_path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def load_benchmark_index(
    benchmarks: list[str],
    extra_jsonl_paths: list[Path] | None = None,
    ngram_size: int = 13,
) -> BenchmarkContaminationIndex:
    """Populate the decontamination index from Hugging Face test splits + optional local JSONLs."""
    import datasets

    index = BenchmarkContaminationIndex(ngram_size=ngram_size)
    for bench in benchmarks:
        if bench not in BENCHMARK_HF_SPECS:
            logger.warning("Unknown benchmark %r skipped.", bench)
            continue
        hf_path, subset, split, fields = BENCHMARK_HF_SPECS[bench]
        logger.info("Indexing benchmark %s (%s [%s])...", bench, hf_path, split)
        try:
            ds_iter: Any = (
                datasets.load_dataset(hf_path, subset, split=split)
                if subset is not None
                else datasets.load_dataset(hf_path, split=split)
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "datasets.load_dataset failed for %s (%s); falling back to raw JSONL files.",
                hf_path,
                exc,
            )
            ds_iter = _iter_hf_repo_jsonl_rows(hf_path)

        for row in ds_iter:
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


def _iter_clean_domain_conversations(
    domain: str,
    index: BenchmarkContaminationIndex,
    seed: int = 42,
) -> Iterator[tuple[list[dict[str, str]], int]]:
    """Yield ``(clean_conversation, cumulative_rejected_count)`` from the HF training split."""
    import datasets

    hf_path, subset, split = DOMAIN_HF_SPECS[domain]
    ds = (
        datasets.load_dataset(hf_path, subset, split=split, streaming=True)
        if subset is not None
        else datasets.load_dataset(hf_path, split=split, streaming=True)
    )
    if isinstance(ds, datasets.IterableDataset):
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    elif hasattr(ds, "shuffle"):
        ds = ds.shuffle(seed=seed)

    rejected = 0
    for row in ds:
        conv = normalize_domain_row(domain, row)
        if conv is None:
            continue
        contaminated, _reason = index.is_contaminated(conv)
        if contaminated:
            rejected += 1
            continue
        yield conv, rejected


def collect_decontaminated_domain_samples(
    domain: str,
    target_count: int,
    index: BenchmarkContaminationIndex,
    seed: int = 42,
) -> tuple[list[dict], int]:
    """Stream a training split from HF Hub and retain ``target_count`` decontaminated conversations."""
    if target_count <= 0:
        return [], 0
    logger.info("Streaming %s for %d clean samples...", domain, target_count)
    accepted: list[dict] = []
    rejected_leaks = 0
    for conv, rej in _iter_clean_domain_conversations(domain, index, seed=seed):
        rejected_leaks = rej
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


async def _post_chat_completion_async(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """POST one chat-completion request with ``return_token_ids=True`` for exact boundary masks."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "return_token_ids": True,
    }
    if temperature == 0.0:
        payload["top_p"] = 1.0

    url = f"{endpoint.rstrip('/')}/chat/completions"
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.post(url, json=payload, timeout=timeout) as resp:
        if not resp.ok:
            body = (await resp.text())[:400]
            raise RuntimeError(f"HTTP {resp.status} from {url}: {body}")
        return await resp.json()


async def regenerate_conversation_turns_on_policy(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    conv: list[dict[str, str]],
    domain: str,
    temperature: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Regenerate assistant turns on-policy and return one boundary row per assistant turn."""
    context: list[dict[str, str]] = []
    rows_out: list[dict[str, Any]] = []
    async with sem:
        for turn in conv:
            if turn["role"] != "assistant":
                context.append(turn)
                continue
            try:
                data = await _post_chat_completion_async(
                    session,
                    endpoint,
                    model,
                    list(context),
                    temperature,
                    max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("On-policy generation failed (%s); truncating conversation.", exc)
                break

            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = str(msg.get("content") or "").strip()
            reasoning = str(msg.get("reasoning_content") or "").strip()
            if not content and not reasoning:
                break

            assistant_turn = {"role": "assistant", "content": content or reasoning}
            turn_conv = [*context, assistant_turn]
            check_turns = list(turn_conv)
            if reasoning and content:
                check_turns.append({"role": "assistant", "content": reasoning})

            row: dict[str, Any] = {
                "conversations": turn_conv,
                "domain": domain,
                "temperature": temperature,
                "_decontam_turns": check_turns,
            }
            prompt_ids = data.get("prompt_token_ids")
            comp_ids = choice.get("token_ids")
            if prompt_ids and comp_ids:
                row["input_ids"] = [*prompt_ids, *comp_ids]
                row["loss_mask"] = [0] * len(prompt_ids) + [1] * len(comp_ids)

            rows_out.append(row)
            context.append(assistant_turn)
    return rows_out


async def _collect_and_regenerate_domain_async(
    domain: str,
    target_count: int,
    index: BenchmarkContaminationIndex,
    endpoint: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    part_path: Path,
    resume: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    """Stream, regenerate on-policy, decontaminate post-gen, and flush until exact ``target_count`` rows."""
    if target_count <= 0:
        return [], 0

    accepted: list[dict[str, Any]] = []
    if resume and part_path.exists():
        with part_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    accepted.append(json.loads(line))
                if len(accepted) >= target_count:
                    break
        if len(accepted) >= target_count:
            logger.info(
                "Domain %s: resumed %d/%d completed rows from %s.",
                domain,
                len(accepted),
                target_count,
                part_path,
            )
            return accepted[:target_count], 0

    part_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (resume and accepted) else "w"
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=max(16, concurrency * 2))
    stream_iter = _iter_clean_domain_conversations(domain, index, seed=seed)

    # Skip already-resumed count slots to reduce duplicate prompts on resume
    for _ in range(len(accepted)):
        if next(stream_iter, None) is None:
            break

    total_rejected = 0
    stream_rej = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        with part_path.open(mode, encoding="utf-8", newline="\n") as out_fh:
            while len(accepted) < target_count:
                needed = target_count - len(accepted)
                batch_size = min(max(concurrency * 2, 8), max(needed, 8))
                candidates: list[tuple[list[dict[str, str]], float]] = []
                for idx in range(batch_size):
                    nxt = next(stream_iter, None)
                    if nxt is None:
                        break
                    conv, stream_rej = nxt
                    temp = assign_sampling_temperature(domain, len(accepted) + idx)
                    candidates.append((conv, temp))
                if not candidates:
                    break

                tasks = [
                    regenerate_conversation_turns_on_policy(
                        sem=sem,
                        session=session,
                        endpoint=endpoint,
                        model=model,
                        conv=conv,
                        domain=domain,
                        temperature=temp,
                        max_tokens=max_tokens,
                    )
                    for conv, temp in candidates
                ]
                batches = await asyncio.gather(*tasks)
                for turn_rows in batches:
                    for row in turn_rows:
                        check_turns = row.pop("_decontam_turns", row["conversations"])
                        contaminated, _ = index.is_contaminated(check_turns)
                        if contaminated:
                            total_rejected += 1
                            continue
                        # Reassign exact stratified temperature slot for accepted index
                        row["temperature"] = assign_sampling_temperature(
                            domain, len(accepted)
                        )
                        accepted.append(row)
                        out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        if len(accepted) >= target_count:
                            break
                    if len(accepted) >= target_count:
                        break
                out_fh.flush()

    total_rejected += stream_rej
    logger.info(
        "Domain %s: completed %d/%d on-policy rows (%d contaminated candidates rejected).",
        domain,
        len(accepted),
        target_count,
        total_rejected,
    )
    return accepted, total_rejected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the decontaminated 100k multi-domain corpus for KvN-DSpark Phase 2."
    )
    parser.add_argument("--verifier-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--ultrachat-samples", type=int, default=45000)
    parser.add_argument("--openmath-samples", type=int, default=32000)
    parser.add_argument("--magicoder-samples", type=int, default=23000)
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
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume from per-domain partial JSONL files if interrupted.",
    )
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
        if count <= 0:
            continue
        if args.vllm_endpoint:
            part_path = args.output_jsonl.with_name(
                f"{args.output_jsonl.stem}.{domain}.part.jsonl"
            )
            logger.info(
                "Collecting & regenerating %d %s rows on-policy via %s...",
                count,
                domain,
                args.vllm_endpoint,
            )
            recs, n_rej = asyncio.run(
                _collect_and_regenerate_domain_async(
                    domain=domain,
                    target_count=count,
                    index=index,
                    endpoint=args.vllm_endpoint,
                    model=args.verifier_model,
                    concurrency=args.concurrency,
                    max_tokens=args.max_completion_tokens,
                    part_path=part_path,
                    resume=bool(args.resume),
                    seed=args.seed,
                )
            )
        else:
            recs, n_rej = collect_decontaminated_domain_samples(
                domain=domain, target_count=count, index=index, seed=args.seed
            )

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
        logger.info(
            "Wrote %d held-out validation conversations to %s",
            len(val_records),
            args.val_output_jsonl,
        )
    else:
        train_records = all_records

    tmp_out = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with tmp_out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in train_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_out.replace(args.output_jsonl)

    manifest_path = args.output_jsonl.with_suffix(".manifest.json")
    manifest = {
        "verifier_model": args.verifier_model,
        "seed": args.seed,
        "ngram_size": args.ngram_size,
        "excluded_benchmarks": args.exclude_benchmarks,
        "on_policy_regenerated": bool(args.vllm_endpoint),
        "pretokenized_boundary_rows": bool(
            train_records and "input_ids" in train_records[0]
        ),
        "domain_counts": {
            d: sum(1 for r in all_records if r.get("domain") == d)
            for d in domain_targets
        },
        "total_written_train": len(train_records),
        "total_written_val": (
            len(all_records) - len(train_records) if args.val_output_jsonl else 0
        ),
        "contaminated_candidates_rejected": total_rejected,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Completed mix build: %s (manifest: %s)", args.output_jsonl, manifest_path
    )


if __name__ == "__main__":
    main()
