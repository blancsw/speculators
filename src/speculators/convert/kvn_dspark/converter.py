"""KvN-DSpark checkpoint exporter and vLLM V1 deployment packager.

Exports a trained ``KvNDSparkDraftModel`` training checkpoint into a self-contained
vLLM-ready checkpoint directory, optionally applying zero-shot dyadic Group-LASSO
mode pruning (``M_train -> M_infer``), enforcing ``kvn_state_dtype="float32"``, and
writing ``vllm_speculative_config.json`` for direct ``vllm serve`` deployment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from speculators.models.kvn_dspark import KvNDSparkDraftModel
from speculators.vllm_integration.plugin import PROPOSER_IMPORT_PATH

__all__ = ["KvNDSparkConverter"]


class KvNDSparkConverter:
    """Package and optionally prune a ``KvNDSparkDraftModel`` checkpoint for vLLM V1."""

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        base_model: str | None = None,
        validate: bool = True,
        *,
        prune_modes: int | None = None,
        load_verifier_weights: bool = False,
        t2d_path: str | Path | None = None,
        d2t_path: str | Path | None = None,
    ) -> Path:
        """Export ``input_path`` to a vLLM-ready checkpoint at ``output_path``."""
        in_dir = Path(input_path)
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Exporting KvN-DSpark checkpoint from {in_dir} -> {out_dir}")

        loaded = KvNDSparkDraftModel.from_pretrained(in_dir)
        if not isinstance(loaded, KvNDSparkDraftModel):
            raise TypeError(
                f"Expected KvNDSparkDraftModel, got {type(loaded).__name__}."
            )

        if base_model:
            loaded.config.speculators_config.verifier.name_or_path = base_model

        if t2d_path is not None and d2t_path is not None:
            t2d = torch.from_numpy(np.load(Path(t2d_path))).bool()
            d2t = torch.from_numpy(np.load(Path(d2t_path))).long()
            loaded.load_vocab_mappings(t2d=t2d, d2t=d2t)

        if load_verifier_weights:
            loaded.load_verifier_weights()

        if prune_modes is not None and int(prune_modes) < int(
            loaded.config.kvn_n_modes
        ):
            loaded.prune_kvn_modes(int(prune_modes))

        loaded.config.kvn_state_dtype = "float32"
        loaded.eval()
        loaded.save_pretrained(out_dir)

        manifest = self._build_vllm_manifest(loaded, out_dir)
        manifest_path = out_dir / "vllm_speculative_config.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if validate:
            self._validate(out_dir)
        return out_dir

    @staticmethod
    def _build_vllm_manifest(model: KvNDSparkDraftModel, out_dir: Path) -> dict:
        cfg = model.config
        num_spec = (
            int(cfg.block_size)
            if cfg.sample_from_anchor
            else max(1, int(cfg.block_size) - 1)
        )
        if cfg.kvn_backbone_type == "swa":
            return {
                "method": "dspark",
                "model": str(out_dir.resolve()),
                "num_speculative_tokens": num_spec,
            }
        return {
            "model": str(out_dir.resolve()),
            "num_speculative_tokens": num_spec,
            "custom_class_fallback": {
                "method": "custom_class",
                "model": PROPOSER_IMPORT_PATH,
                "num_speculative_tokens": num_spec,
            },
        }

    @staticmethod
    def _validate(saved_path: Path) -> None:
        reloaded = KvNDSparkDraftModel.from_pretrained(saved_path)
        if not isinstance(reloaded, KvNDSparkDraftModel):
            raise RuntimeError(
                f"Reloaded checkpoint had unexpected class {type(reloaded).__name__}."
            )
