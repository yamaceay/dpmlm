"""Utility helpers for assembling DPMLM configuration dictionaries."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger(__name__)


DPMLM_GENERIC_KEYS = {"device", "seed", "verbose"}
DPMLM_RUNTIME_KEYS = {"input_text"}
DPMLM_MODEL_KEYS = {
    "model",
    "alpha",
    "clip_min",
    "clip_max",
    "use_treebank_tokenizer",
    "k_candidates",
    "use_temperature",
    "process_pii_only",
    "add_probability",
    "delete_probability",
    "risk_pipeline",
    "annotator",
    "pii_threshold",
    "min_weight",
    "maintain_expected_noise",
    "clip_contribution",
    "explainability_mode",
    "explainability_label",
    "mask_text",
    "shap_silent",
    "shap_batch_size",
}
DPMLM_SPECIAL_MODEL_KEYS = {
    "annotator_path",
    "risk_model",
    "risk_task",
    "risk_max_length",
    "risk_pipeline_kwargs",
    "model_config",
}
MODEL_CONFIG_KEYS = {"model_name", "max_sequence_length", "truncation", "padding", "cache_dir"}
RISK_DEFAULT_MAX_LENGTH = 512


def coerce_dpmlm_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Coerce legacy DPMLM config mappings into generic/model/runtime buckets."""

    structured: Dict[str, Dict[str, Any]] = {"generic": {}, "model": {}, "runtime": {}}
    if not raw:
        return structured

    if not isinstance(raw, dict):
        raise ValueError("DPMLM configuration must be provided as a mapping.")

    for section in ("generic", "model", "runtime"):
        value = raw.get(section)
        if isinstance(value, dict):
            structured[section].update(value)

    if isinstance(raw.get("dpmlm"), dict):
        structured["model"].update(raw["dpmlm"])
    if isinstance(raw.get("dpmlm_config"), dict):
        structured["model"].update(raw["dpmlm_config"])
    if isinstance(raw.get("model_config"), dict):
        structured["model"].setdefault("model", {}).update(raw["model_config"])

    for risk_key in ("risk", "risk_settings", "risk_config"):
        risk_value = raw.get(risk_key)
        if isinstance(risk_value, dict):
            structured["model"].update(risk_value)

    leftover_keys = {
        key
        for key in raw
        if key
        not in {
            "generic",
            "model",
            "runtime",
            "dpmlm",
            "dpmlm_config",
            "risk",
            "risk_settings",
            "risk_config",
            "model_config",
        }
    }
    for key in leftover_keys:
        structured["model"][key] = raw[key]

    return structured


def resolve_device(preference: str) -> str:
    """Resolve a device preference string to an available backend."""
    if preference != "auto":
        return preference

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_risk_pipeline(
    model_name: str,
    task: str,
    device_pref: str,
    max_length: int = RISK_DEFAULT_MAX_LENGTH,
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    """Instantiate a Hugging Face pipeline for risk-aware scoring."""
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Transformers is required for risk-aware scoring.") from exc

    kwargs = dict(extra_kwargs or {})
    kwargs.setdefault("top_k", None)
    kwargs.setdefault("device", resolve_device(device_pref))
    kwargs.setdefault("truncation", True)
    kwargs.setdefault("max_length", max_length)

    return hf_pipeline(task, model=model_name, tokenizer=model_name, **kwargs)


def prepare_dpmlm_model_config(
    model_cfg: Dict[str, Any],
    generic_cfg: Dict[str, Any],
    *,
    annotator_loader: Optional[Callable[[str], Optional[Any]]] = None,
    pipeline_loader: Optional[
        Callable[[str, str, str, int, Optional[Dict[str, Any]]], Any]
    ] = None,
) -> Dict[str, Any]:
    """Normalise model config and instantiate dynamic resources."""

    payload: Dict[str, Any] = dict(model_cfg or {})

    transformer_config = dict(payload.pop("model", {}))
    legacy_transformer = payload.pop("model_config", None)
    if isinstance(legacy_transformer, dict):
        transformer_config.update(legacy_transformer)

    for key in list(payload.keys()):
        if key in MODEL_CONFIG_KEYS:
            transformer_config.setdefault(key, payload.pop(key))

    transformer_config.setdefault("model_name", "roberta-base")

    annotator_path = payload.pop("annotator_path", None)
    if annotator_path and annotator_loader is not None:
        annotator = annotator_loader(annotator_path)
        if annotator is not None:
            payload["annotator"] = annotator
            payload.setdefault("process_pii_only", True)
        else:
            logger.warning("Disabling process_pii_only because annotator load failed.")
            payload["process_pii_only"] = False

    risk_model = payload.pop("risk_model", None)
    risk_task = payload.pop("risk_task", "text-classification")
    risk_max_length = int(payload.pop("risk_max_length", RISK_DEFAULT_MAX_LENGTH))
    risk_kwargs = payload.pop("risk_pipeline_kwargs", None)
    if risk_kwargs is not None and not isinstance(risk_kwargs, dict):
        raise ValueError("risk_pipeline_kwargs must be a mapping if provided.")

    if risk_model:
        loader = pipeline_loader or build_risk_pipeline
        risk_pipeline = loader(
            risk_model,
            risk_task,
            generic_cfg.get("device", "auto"),
            risk_max_length,
            risk_kwargs,
        )
        payload["risk_pipeline"] = risk_pipeline
        tokenizer = getattr(risk_pipeline, "tokenizer", None)
        mask_token = getattr(tokenizer, "mask_token", None)
        if mask_token and "mask_text" not in payload:
            payload["mask_text"] = mask_token

    if payload.get("process_pii_only") and payload.get("annotator") is None:
        logger.warning("process_pii_only requested but no annotator configured; disabling it.")
        payload["process_pii_only"] = False

    mode = (payload.get("explainability_mode") or "uniform").lower()
    if mode in {"greedy", "shap"} and payload.get("risk_pipeline") is None:
        raise ValueError(
            "DPMLM explainability mode 'greedy' or 'shap' requires risk_model in the configuration."
        )

    filtered_payload: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in DPMLM_MODEL_KEYS:
            filtered_payload[key] = value
        elif key not in DPMLM_SPECIAL_MODEL_KEYS:
            logger.debug("Ignoring unsupported DPMLM model parameter: %s", key)

    filtered_payload["model"] = transformer_config
    return filtered_payload
