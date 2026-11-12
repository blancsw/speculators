#!/usr/bin/env python3
"""Build the decontaminated 100k multi-domain corpus for KvN-DSpark.

Implements Phase B of ``kvn_rff_rnn_mtp_train_phase_2.md``:
1. Pulls strictly from **training** splits of three complementary domains (100,000):
   - ``HuggingFaceH4/ultrachat_200k`` (``train_sft``, 45k): General dialogue
   - ``nvidia/OpenMathInstruct-2`` (``train``, 32k): Long-horizon mathematical CoT
   - ``ise-uiuc/Magicoder-OSS-Instruct-75K`` (``train``, 23k): Code generation
2. Enforces **strict SHA-256 + normalized 13-gram decontamination** against downstream
   evaluation benchmarks (MT-Bench, GSM8K test, MATH-500, HumanEval, MBPP,
   LiveCodeBench, plus any custom JSONL), with automatic JSONL fallback for Hugging
   Face repos whose legacy Python loading scripts are deprecated under ``datasets>=3``.
3. Assigns stratified sampling temperatures matching the Phase 2 protocol:
   - UltraChat: 50% T=0.0 / 50% T=0.7
   - OpenMathInstruct-2: 50% T=0.0 / 50% T=0.6
   - Magicoder: 60% T=0.0 / 40% T=0.6
4. Optionally regenerates assistant turns **on-policy** against a live ``vLLM`` server
   (``--vllm-endpoint``) using pooled ``aiohttp`` + ``"return_token_ids": True`` to emit
   **token-exact pretokenized ``(input_ids, loss_mask)`` boundary rows**, re-verifies
   generated completions against 13-gram benchmark overlap, and prepares an aligned
   conversation-disjoint ``ArrowDataset`` directory via ``--prepare-arrow-dir``.
"""

# ruff: noqa: ASYNC240, C901, PLC0415, PLR2004

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger("build_decontaminated_mix")

_WORD_RE = re.compile(r"[a-z0-9]+")

# Hugging Face evaluation benchmark specifications used solely to build the exclusion
# filter.
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
    """Normalize text to lowercase alphanumeric tokens (NFKC) for SHA-256 & n-gram
    matching.
    """
    norm = unicodedata.normalize("NFKC", text).lower()
    return _WORD_RE.findall(norm)


def _hash_tokens(tokens: list[str] | tuple[str, ...]) -> bytes:
    joined = " ".join(tokens).encode("utf-8")
    return hashlib.sha256(joined).digest()[:16]


@dataclass
class BenchmarkContaminationIndex:
    """Exact normalized-prompt SHA-256 + 13-gram index over evaluation benchmarks."""

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
        """Return (True, reason) if any turn collides by SHA-256 or n-gram overlap."""
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
    """Convert raw domain dataset record into canonical ``[{'role': ..., 'content':
    ...}]``.
    """
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


def compute_conversation_id(domain: str, conv: list[dict[str, str]]) -> str:
    """Compute a deterministic 16-hex SHA-256 ID for a source conversation."""
    prompt_turns = [
        {"role": t.get("role", ""), "content": t.get("content", "")}
        for t in conv
        if t.get("role") != "assistant"
    ]
    basis = prompt_turns if prompt_turns else conv
    payload = json.dumps(
        {"domain": domain, "turns": basis},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def is_truncated_or_unclosed_think(
    choice: dict[str, Any],
    content: str,
    reasoning: str,
) -> bool:
    """Return True if generation hit max_tokens or left an unclosed <think> block."""
    finish_reason = str(choice.get("finish_reason") or "").strip().lower()
    if finish_reason == "length":
        return True
    if reasoning and not content:
        return True
    if "<think>" in content:
        if content.count("<think>") > content.count("</think>"):
            return True
        if content.rfind("<think>") > content.rfind("</think>"):
            return True
    return False


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
    """Populate the decontamination index from Hugging Face test splits + optional local
    JSONLs.
    """
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
                "datasets.load_dataset failed for %s (%s); falling back to JSONL.",
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
    """Yield ``(clean_conversation, cumulative_rejected_count)`` from the HF training
    split.
    """
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
    """Stream a training split from HF Hub and retain ``target_count`` decontaminated
    conversations.
    """
    if target_count <= 0:
        return [], 0
    logger.info("Streaming %s for %d clean samples...", domain, target_count)
    accepted: list[dict] = []
    seen_conv_ids: set[str] = set()
    rejected_leaks = 0
    for conv, rej in _iter_clean_domain_conversations(domain, index, seed=seed):
        rejected_leaks = rej
        conv_id = compute_conversation_id(domain, conv)
        if conv_id in seen_conv_ids:
            continue
        seen_conv_ids.add(conv_id)
        temp = assign_sampling_temperature(domain, len(accepted))
        accepted.append(
            {
                "conversation_id": conv_id,
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
    """POST one chat-completion request with return_token_ids=True."""
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
    conv_id: str | None = None,
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Regenerate assistant turns on-policy and return one row per assistant turn."""
    resolved_conv_id = conv_id or compute_conversation_id(domain, conv)
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
                logger.warning(
                    "On-policy generation failed (%s); truncating conversation.", exc
                )
                break

            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = str(msg.get("content") or "").strip()
            reasoning = str(msg.get("reasoning_content") or "").strip()
            if not content and not reasoning:
                break

            if is_truncated_or_unclosed_think(choice, content, reasoning):
                if stats is not None:
                    stats["truncated_rejected"] = stats.get("truncated_rejected", 0) + 1
                break

            assistant_turn = {"role": "assistant", "content": content or reasoning}
            turn_conv = [*context, assistant_turn]
            check_turns = list(turn_conv)
            if reasoning and content:
                check_turns.append({"role": "assistant", "content": reasoning})

            row: dict[str, Any] = {
                "conversation_id": resolved_conv_id,
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


def _group_records_by_conversation_id(
    records: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group sibling turn-rows by conversation_id in first-seen conversation order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for idx, rec in enumerate(records):
        cid = rec.get("conversation_id")
        key = str(cid) if cid is not None and str(cid) != "" else f"__legacy_{idx}"
        groups.setdefault(key, []).append(rec)
    return list(groups.values())


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
    stats_out: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Stream, regenerate on-policy, decontaminate post-gen, and flush until
    ``target_count`` unique conversations.
    """
    if target_count <= 0:
        return [], 0

    accepted: list[dict[str, Any]] = []
    accepted_conv_ids: set[str] = set()
    if resume and part_path.exists():
        raw_resumed: list[dict[str, Any]] = []
        with part_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    raw_resumed.append(json.loads(line))
        resumed_groups = _group_records_by_conversation_id(raw_resumed)
        for grp_idx, grp in enumerate(resumed_groups[:target_count]):
            cid = str(
                grp[0].get("conversation_id")
                or compute_conversation_id(domain, grp[0].get("conversations", []))
                or f"__legacy_{grp_idx}"
            )
            for r in grp:
                r.setdefault("conversation_id", cid)
                accepted.append(r)
            accepted_conv_ids.add(cid)
        if len(accepted_conv_ids) >= target_count:
            logger.info(
                "Domain %s: resumed %d/%d completed conversations (%d rows) from %s.",
                domain,
                len(accepted_conv_ids),
                target_count,
                len(accepted),
                part_path,
            )
            return accepted, 0

    part_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (resume and accepted) else "w"
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=max(16, concurrency * 2))
    stream_iter = _iter_clean_domain_conversations(domain, index, seed=seed)

    # Skip already-resumed unique conversations to avoid duplicate prompts on resume
    for _ in range(len(accepted_conv_ids)):
        if next(stream_iter, None) is None:
            break

    total_rejected = 0
    stream_rej = 0
    trunc_stats: dict[str, int] = {"truncated_rejected": 0}

    async with aiohttp.ClientSession(connector=connector) as session:
        with part_path.open(mode, encoding="utf-8", newline="\n") as out_fh:
            while len(accepted_conv_ids) < target_count:
                needed = target_count - len(accepted_conv_ids)
                batch_size = min(max(concurrency * 2, 8), max(needed, 8))
                candidates: list[tuple[list[dict[str, str]], float, str]] = []
                for idx in range(batch_size):
                    nxt = next(stream_iter, None)
                    if nxt is None:
                        break
                    conv, stream_rej = nxt
                    cid = compute_conversation_id(domain, conv)
                    if cid in accepted_conv_ids:
                        continue
                    temp = assign_sampling_temperature(
                        domain, len(accepted_conv_ids) + idx
                    )
                    candidates.append((conv, temp, cid))
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
                        conv_id=cid,
                        stats=trunc_stats,
                    )
                    for conv, temp, cid in candidates
                ]
                batches = await asyncio.gather(*tasks)
                for turn_rows in batches:
                    if not turn_rows:
                        continue
                    conv_id = str(turn_rows[0]["conversation_id"])
                    if conv_id in accepted_conv_ids:
                        continue
                    clean_turn_rows: list[dict[str, Any]] = []
                    for row in turn_rows:
                        check_turns = row.pop("_decontam_turns", row["conversations"])
                        contaminated, _ = index.is_contaminated(check_turns)
                        if contaminated:
                            total_rejected += 1
                            # Stop accepting later turns whose context includes a leaked
                            # turn
                            break
                        clean_turn_rows.append(row)

                    if not clean_turn_rows:
                        continue

                    conv_temp = assign_sampling_temperature(
                        domain, len(accepted_conv_ids)
                    )
                    for row in clean_turn_rows:
                        row["temperature"] = conv_temp
                        row["conversation_id"] = conv_id
                        accepted.append(row)
                        out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    accepted_conv_ids.add(conv_id)
                    if len(accepted_conv_ids) >= target_count:
                        break
                out_fh.flush()

    if stats_out is not None:
        stats_out["truncated_rejected"] = (
            stats_out.get("truncated_rejected", 0) + trunc_stats["truncated_rejected"]
        )
    total_rejected += stream_rej
    logger.info(
        "Domain %s: completed %d/%d unique on-policy conversations "
        "(%d turn-rows, %d contaminated rejected, %d truncated rejected).",
        domain,
        len(accepted_conv_ids),
        target_count,
        len(accepted),
        total_rejected,
        trunc_stats["truncated_rejected"],
    )
    return accepted, total_rejected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the decontaminated 100k multi-domain corpus for KvN-DSpark."
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
        help="Optional live vLLM OpenAI endpoint for on-policy regeneration.",
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
    parser.add_argument(
        "--prepare-arrow-dir",
        type=Path,
        default=None,
        help="Optional Arrow dir (conversation-disjoint split + train token_freq.pt).",
    )
    parser.add_argument(
        "--prepare-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip JSONL generation and build --prepare-arrow-dir from --output-jsonl.",
    )
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--train-data-ratio", type=float, default=0.95)
    parser.add_argument("--token-freq-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def prepare_conversation_disjoint_arrow_dataset(
    source: Any,
    output_arrow_dir: Path | str,
    *,
    seq_length: int = 8192,
    train_ratio: float = 0.95,
    seed: int = 42,
    token_freq_path: Path | str | None = None,
    minimum_valid_tokens: int = 1,
    overwrite: bool = False,
) -> Any:
    """Write a conversation-grouped Arrow dataset aligned to ArrowDataset's split."""
    from datasets import Dataset as HFDataset
    from datasets import load_dataset

    from speculators.train.vocab_mapping import save_token_frequency_distribution

    out_dir = Path(output_arrow_dir)
    if out_dir.exists() and not overwrite and list(out_dir.glob("*.arrow")):
        logger.info("Arrow dataset already exists at %s; skipping.", out_dir)
        return None

    raw_ds = (
        source
        if isinstance(source, HFDataset)
        else load_dataset("json", data_files=str(source), split="train")
    )
    raw_rows = raw_ds.with_format(None).to_list()
    groups = _group_records_by_conversation_id(raw_rows)
    random.Random(seed).shuffle(groups)

    kept_rows: list[dict[str, Any]] = []
    for g_idx, grp in enumerate(groups):
        for rec in grp:
            ids = list(rec["input_ids"])[:seq_length]
            mask = [bool(m) for m in list(rec["loss_mask"])[:seq_length]]
            if sum(mask) < minimum_valid_tokens:
                continue
            cid = str(rec.get("conversation_id") or f"conv_{g_idx}")
            kept_rows.append(
                {
                    "conversation_id": cid,
                    "input_ids": ids,
                    "loss_mask": mask,
                    "seq_len": len(ids),
                }
            )

    if 0.0 < train_ratio < 1.0:
        while (
            len(kept_rows) >= 2
            and 0 < (s := int(len(kept_rows) * train_ratio)) < len(kept_rows)
            and kept_rows[s - 1]["conversation_id"] == kept_rows[s]["conversation_id"]
        ):
            kept_rows.pop()

    if not kept_rows:
        raise ValueError("No valid pretokenized rows remained after filtering.")

    ds = HFDataset.from_list(kept_rows)
    ds.set_format(type="torch")
    out_dir.mkdir(parents=True, exist_ok=True)

    freq_out = (
        Path(token_freq_path)
        if token_freq_path is not None
        else out_dir / "token_freq.pt"
    )
    split_idx = int(len(ds) * train_ratio) if 0.0 < train_ratio < 1.0 else len(ds)
    save_token_frequency_distribution(
        dataset=ds.select(range(max(1, split_idx))),
        output_path=freq_out,
    )
    ds.save_to_disk(str(out_dir))
    logger.info(
        "Saved conversation-disjoint Arrow dataset (%d rows, train_split=%d) to %s",
        len(ds),
        split_idx,
        out_dir,
    )
    return ds


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    args = parse_args(argv)
    if args.prepare_only:
        if args.prepare_arrow_dir is None:
            raise ValueError("--prepare-only requires --prepare-arrow-dir.")
        prepare_conversation_disjoint_arrow_dataset(
            args.output_jsonl,
            args.prepare_arrow_dir,
            seq_length=args.seq_length,
            train_ratio=args.train_data_ratio,
            seed=args.seed,
            token_freq_path=args.token_freq_path,
        )
        return
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
    gen_stats: dict[str, int] = {"truncated_rejected": 0}
    for domain, count in domain_targets.items():
        if count <= 0:
            continue
        if args.vllm_endpoint:
            part_path = args.output_jsonl.with_name(
                f"{args.output_jsonl.stem}.{domain}.part.jsonl"
            )
            logger.info(
                "Collecting & regenerating %d %s conversations on-policy via %s...",
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
                    stats_out=gen_stats,
                )
            )
        else:
            recs, n_rej = collect_decontaminated_domain_samples(
                domain=domain, target_count=count, index=index, seed=args.seed
            )

        all_records.extend(recs)
        total_rejected += n_rej

    # Group by conversation_id BEFORE shuffling or train/val splitting so 0.0%
    # of held-out validation conversations share sibling prefixes with train.
    conv_groups = _group_records_by_conversation_id(all_records)
    rng.shuffle(conv_groups)
    all_records = [rec for grp in conv_groups for rec in grp]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if (
        args.val_output_jsonl is not None
        and args.val_samples > 0
        and len(conv_groups) >= 2
    ):
        val_conv_count = min(args.val_samples, len(conv_groups) - 1)
        val_groups = conv_groups[:val_conv_count]
        train_groups = conv_groups[val_conv_count:]
        val_records = [rec for grp in val_groups for rec in grp]
        train_records = [rec for grp in train_groups for rec in grp]
        args.val_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.val_output_jsonl.open("w", encoding="utf-8", newline="\n") as fh:
            for rec in val_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(
            "Wrote %d held-out validation rows (%d unique conversations) to %s",
            len(val_records),
            len(val_groups),
            args.val_output_jsonl,
        )
    else:
        val_groups = []
        train_groups = conv_groups
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
        "domain_conversation_counts": {
            d: len(
                {
                    r.get("conversation_id")
                    for r in all_records
                    if r.get("domain") == d and r.get("conversation_id")
                }
            )
            for d in domain_targets
        },
        "total_written_train": len(train_records),
        "total_written_val": (
            len(all_records) - len(train_records) if args.val_output_jsonl else 0
        ),
        "unique_conversations_train": len(train_groups),
        "unique_conversations_val": len(val_groups),
        "contaminated_candidates_rejected": total_rejected,
        "truncated_generations_rejected": gen_stats["truncated_rejected"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Completed mix build: %s (manifest: %s)", args.output_jsonl, manifest_path
    )
    if args.prepare_arrow_dir is not None:
        prepare_conversation_disjoint_arrow_dataset(
            args.output_jsonl,
            args.prepare_arrow_dir,
            seq_length=args.seq_length,
            train_ratio=args.train_data_ratio,
            seed=args.seed,
            token_freq_path=args.token_freq_path,
        )


if __name__ == "__main__":
    main()
