"""
Core DPMLM components with high-level, type-safe interfaces.

This module provides the main DPMLM mechanism implementation following
the same design patterns as the PII module.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import string
import numpy as np
import torch
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from collections import Counter
from abc import ABC, abstractmethod

from dpmlm.interfaces import DPMechanism, PrivacyResult, TokenSpan, DPTokenizer
from dpmlm.config import DPMLMConfig
from dpmlm.resources import DPMLMModelManager

logger = logging.getLogger(__name__)


class DPMLMTokenizer:
    """High-level tokenizer for DPMLM with span information."""
    
    def __init__(self, use_treebank: bool = True):
        self.use_treebank = use_treebank
        self.tokenizer = TreebankWordTokenizer() if use_treebank else None
        self.detokenizer = TreebankWordDetokenizer()
        
    def tokenize_with_spans(self, text: str) -> List[TokenSpan]:
        """Tokenize text and return token spans with position information."""
        if self.use_treebank:
            spans = list(self.tokenizer.span_tokenize(text))
            return [
                TokenSpan(
                    text=text[start:end],
                    start=start,
                    end=end,
                    is_critical=False
                )
                for start, end in spans
            ]
        else:
            
            import nltk
            tokens = nltk.word_tokenize(text)
            spans = []
            pos = 0
            for token in tokens:
                start = text.find(token, pos)
                if start >= 0:
                    end = start + len(token)
                    spans.append(TokenSpan(
                        text=token,
                        start=start,
                        end=end,
                        is_critical=False
                    ))
                    pos = end
            return spans
    
    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        return self.detokenizer.detokenize(tokens)


class DPMLMCore:
    """Core DPMLM functionality with clean interface."""
    
    def __init__(self, config: DPMLMConfig):
        self.config = config
        self.tokenizer_wrapper = DPMLMTokenizer(config.use_treebank_tokenizer)
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.config.validate()
        logger.info("DPMLM configuration validated")
        
    def _get_device(self) -> torch.device:
        """Get appropriate device."""
        if self.config.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)
    
    def _sentence_enum(self, tokens: List[str]) -> List[int]:
        """Enumerate token occurrences."""
        counts = Counter()
        n = []
        for token in tokens:
            counts[token] += 1
            n.append(counts[token])
        return n
    
    def _nth_replace(self, text: str, target: str, replacement: str, occurrence: int) -> str:
        """Replace the nth occurrence of target with replacement."""
        parts = text.split()
        count = 0
        for i, part in enumerate(parts):
            if part == target:
                count += 1
                if count == occurrence:
                    parts[i] = replacement
                    break
        return " ".join(parts)
    
    def _nth_remove(self, text: str, target: str, occurrence: int) -> str:
        """Remove the nth occurrence of target."""
        parts = text.split()
        count = 0
        for i, part in enumerate(parts):
            if part == target:
                count += 1
                if count == occurrence:
                    parts.pop(i)
                    break
        return " ".join(parts)
    
    def _calculate_temperature(self, epsilon: float) -> float:
        """Calculate temperature for differential privacy."""
        return 2 * self.config.sensitivity / epsilon
    
    def _chunk_sequence_if_needed(
        self,
        input_ids: List[int],
        mask_positions: List[int],
    ) -> List[Tuple[List[int], List[int]]]:
        """Return a single truncated window containing the mask token."""
        max_len = self.config.model_config.max_sequence_length

        if len(input_ids) <= max_len:
            return [(input_ids, mask_positions)]

        truncated_ids = input_ids[:max_len]
        adjusted_positions = [pos for pos in mask_positions if pos < max_len]
        if not adjusted_positions and mask_positions:
            adjusted_positions = [min(mask_positions[0], max_len - 1)]

        return [(truncated_ids, adjusted_positions)]


# ---------------------------------------------------------------------------
# Strategy-based DPMLM rewriting
# ---------------------------------------------------------------------------


@dataclass
class DPMLMRewriteSettings:
    """Configuration for risk-aware rewriting behaviour."""

    risk_pipeline: Optional[Any] = None
    annotator: Optional[Any] = None
    pii_threshold: float = 0.0
    min_weight: float = 1e-6
    maintain_expected_noise: bool = True
    top_k: Optional[int] = None
    clip_contribution: Optional[float] = None
    explainability_mode: str = "uniform"
    explainability_label: Optional[Any] = None
    mask_text: str = "<mask>"
    shap_silent: bool = True
    shap_batch_size: int = 1
    summary_top_k: int = 10


class TokenSelectionStrategy(ABC):
    """Defines which token spans should be considered for protection."""

    def __init__(self, annotator: Optional[Any] = None, threshold: float = 0.0):
        self._annotator = annotator
        self.threshold = threshold

    @abstractmethod
    def mark(
        self,
        spans: List[TokenSpan],
        text: str,
        *args,
    ) -> List[TokenSpan]:
        """Mark spans as critical in-place and return the list."""


class AllTokensSelection(TokenSelectionStrategy):
    """Marks every token span as critical."""

    def mark(self, spans: List[TokenSpan], text: str, *args) -> List[TokenSpan]:  # noqa: ARG002
        for span in spans:
            span.is_critical = True
        return spans


class PIITokenSelection(TokenSelectionStrategy):
    """Marks spans intersecting PII predictions as critical."""

    def mark(self, spans: List[TokenSpan], text: str, *args) -> List[TokenSpan]:
        annotator = self._annotator
        if annotator is None:
            raise ValueError(
                "PII token selection requested but no annotator provided; falling back to baseline selection."
            )

        predictions = annotator.predict(text)
        unique_labels = getattr(annotator.labels, "unique_labels", [])

        for span in spans:
            span.is_critical = False

        for prediction in predictions:
            entity_group = prediction.get("entity_group")
            if unique_labels and entity_group not in unique_labels:
                continue
            start, end = prediction.get("start"), prediction.get("end")
            if start is None or end is None:
                continue
            if prediction.get("score") is not None and prediction["score"] < self.threshold:
                continue
            for span in spans:
                if span.start < end and span.end > start:
                    span.is_critical = True
                    span.entity_type = entity_group

        if not any(span.is_critical for span in spans):
            logger.debug("No PII spans detected; defaulting to baseline selection.")
            return AllTokensSelection().mark(spans, text)

        return spans


@dataclass
class ScoringContext:
    risk_settings: DPMLMRewriteSettings
    mechanism: "DPMLMMechanism"


class ScoringStrategy(ABC):
    """Produces contribution scores for a list of critical spans."""

    def __init__(self, name: str = "uniform"):
        self.name = name

    @abstractmethod
    def score(
        self,
        text: str,
        spans: List[TokenSpan],
        context: ScoringContext,
    ) -> np.ndarray:
        """Return raw (unnormalised) contribution scores."""


class UniformScoring(ScoringStrategy):
    def __init__(self) -> None:
        super().__init__(name="uniform")

    def score(self, text: str, spans: List[TokenSpan], context: ScoringContext) -> np.ndarray:  # noqa: ARG002
        if not spans:
            return np.array([], dtype=float)
        return np.ones(len(spans), dtype=float)


class GreedyScoring(ScoringStrategy):
    def __init__(self) -> None:
        super().__init__(name="greedy")

    def score(self, text: str, spans: List[TokenSpan], context: ScoringContext) -> np.ndarray:
        pipeline = context.risk_settings.risk_pipeline
        if pipeline is None or not spans:
            logger.debug("Greedy scoring requested without pipeline; reverting to uniform weights.")
            return np.ones(len(spans), dtype=float)

        label_index = context.mechanism._resolve_label_index(pipeline, text)
        base_prob = context.mechanism._label_probability(pipeline, text, label_index)
        if base_prob is None:
            logger.debug("Failed to obtain base probability for greedy scoring; using uniform weights.")
            return np.ones(len(spans), dtype=float)

        scores: List[float] = []
        mask_text = context.risk_settings.mask_text
        for span in spans:
            masked_text = context.mechanism._mask_span(text, span, mask_text)
            prob = context.mechanism._label_probability(pipeline, masked_text, label_index)
            if prob is None:
                scores.append(0.0)
            else:
                scores.append(float(max(0.0, base_prob - prob)))

        scores_array = np.array(scores, dtype=float)
        if np.allclose(scores_array, 0.0):
            logger.debug("Greedy scores collapsed to zero; returning uniform weights.")
            return np.ones(len(spans), dtype=float)
        return scores_array


class ShapScoring(ScoringStrategy):
    def __init__(self) -> None:
        super().__init__(name="shap")

    def score(self, text: str, spans: List[TokenSpan], context: ScoringContext) -> np.ndarray:
        pipeline = context.risk_settings.risk_pipeline
        if pipeline is None or not spans:
            logger.debug("SHAP scoring requested without pipeline; reverting to uniform weights.")
            return np.ones(len(spans), dtype=float)

        explainer = context.mechanism._ensure_shap_explainer(pipeline)
        if explainer is None:
            logger.debug("Unable to initialise SHAP explainer; using uniform weights.")
            return np.ones(len(spans), dtype=float)

        try:
            shap_values = explainer([text], batch_size=context.risk_settings.shap_batch_size)
        except TypeError:
            shap_values = explainer([text])
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to compute SHAP values (%s); using uniform weights.", exc)
            return np.ones(len(spans), dtype=float)

        if not hasattr(shap_values, "values"):
            logger.debug("SHAP values object missing 'values'; using uniform weights.")
            return np.ones(len(spans), dtype=float)

        label_index = context.mechanism._resolve_label_index(pipeline, text)
        token_weights = shap_values.values[0, :, label_index]
        tokenizer = getattr(pipeline, "tokenizer", None)
        if tokenizer is None:
            logger.debug("Risk pipeline lacks tokenizer; SHAP scoring falls back to uniform weights.")
            return np.ones(len(spans), dtype=float)

        term_to_token = context.mechanism._terms_to_tokens(text, spans, tokenizer)
        scores: List[float] = []
        for token_indices in term_to_token:
            if not token_indices:
                scores.append(0.0)
            else:
                contrib = sum(
                    float(token_weights[idx])
                    for idx in token_indices
                    if 0 <= idx < len(token_weights)
                )
                scores.append(contrib)

        scores_array = np.array(scores, dtype=float)
        if np.allclose(scores_array, 0.0):
            logger.debug("SHAP scores collapsed to zero; defaulting to uniform weights.")
            return np.ones(len(spans), dtype=float)
        return scores_array


@dataclass
class TokenRiskAllocation:
    span: TokenSpan
    score: float
    weight: float
    epsilon: float

    def to_metadata(self, rank: int) -> Dict[str, Any]:
        data = {
            "rank": rank,
            "start": int(self.span.start),
            "end": int(self.span.end),
            "token": self.span.text.replace("\n", " "),
            "score": float(self.score),
            "weight": float(self.weight),
            "epsilon": float(self.epsilon),
        }
        if getattr(self.span, "entity_type", None):
            data["entity_type"] = self.span.entity_type
        return data


class DPMLMMechanism(DPMechanism):
    """High-level DPMLM rewrite mechanism driven by strategies."""

    def __init__(
        self,
        config: DPMLMConfig,
        settings: DPMLMRewriteSettings,
        selection_strategy: Optional[TokenSelectionStrategy] = None,
        scoring_strategy: Optional[ScoringStrategy] = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.core = DPMLMCore(config)
        self.device = self.core._get_device()
        logger.info("DPMLM privatizer initialized on device: %s", self.device)

        self.selection_strategy = selection_strategy or self._build_selection_strategy()
        self.scoring_strategy = scoring_strategy or self._build_scoring_strategy()
        self._shap_explainer = None

    def validate_epsilon(self, epsilon: float) -> bool:
        return epsilon > 0

    def privatize(self, text: str, epsilon: float, **kwargs) -> PrivacyResult:
        if epsilon <= 0:
            raise ValueError(f"Invalid epsilon: {epsilon}")

        spans = self.core.tokenizer_wrapper.tokenize_with_spans(text)
        spans = self.selection_strategy.mark(
            spans,
            text,
        )
        critical_spans = [span for span in spans if span.is_critical]

        context = ScoringContext(
            risk_settings=self.settings,
            mechanism=self,
        )
        scores = self.scoring_strategy.score(text, critical_spans, context)
        if scores.size != len(critical_spans):
            scores = np.ones(len(critical_spans), dtype=float)

        ranking_weights = self._compute_ranking_weights(scores)
        allocations = self._build_allocations(critical_spans, scores, ranking_weights, epsilon)
        allocations.sort(key=lambda alloc: (alloc.weight, alloc.score), reverse=True)

        token_eps_map = {
            (alloc.span.start, alloc.span.end): alloc.epsilon for alloc in allocations
        }

        if not self.validate_epsilon(epsilon):
            raise ValueError(f"Invalid epsilon: {epsilon}")

        use_plus = kwargs.get('plus', False)
        
        if not use_plus:
            kwargs.update({
                'add_probability': 0.0, 
                'delete_probability': 0.0,
            })


        # Get hyperparameters from kwargs with defaults
        add_probability = kwargs.get('add_probability', kwargs.get('ADD_PROB', 0.15))
        delete_probability = kwargs.get('delete_probability', kwargs.get('DEL_PROB', 0.05))
        
        perturbed = 0
        total = 0
        added = 0
        deleted = 0
        result = text
        offset_adjust = 0
        
        with DPMLMModelManager(self.config.model_config, str(self.device)) as (models, tokenizer):
            lm_model, raw_model = models
            
            for i, span in enumerate(spans):
                if not span.is_critical:
                    total += 1
                    continue
                    
                if span.text in string.punctuation:
                    total += 1
                    continue
                
                span_key = (span.start, span.end)
                local_epsilon = token_eps_map.get(span_key) if token_eps_map else epsilon
                if not local_epsilon or local_epsilon <= 0:
                    local_epsilon = epsilon

                adjusted_start = span.start + offset_adjust
                adjusted_end = span.end + offset_adjust
                
                
                if i == len(spans) - 1:
                    delete_prob = 1.0  
                else:
                    delete_prob = np.random.rand()
                
                if delete_prob >= delete_probability:
                    
                    private_token = self._privatize_single_token(
                        text, span.text, local_epsilon, lm_model, raw_model, tokenizer, **kwargs
                    )
                    
                    
                    if span.text and span.text[0].isupper():
                        private_token = private_token.capitalize() if private_token else span.text
                    elif span.text and span.text[0].islower():
                        private_token = private_token.lower() if private_token else span.text
                    
                    
                    result = result[:adjusted_start] + private_token + result[adjusted_end:]
                    offset_adjust += len(private_token) - len(span.text)
                    
                    if private_token != span.text:
                        perturbed += 1
                    total += 1
                    
                    
                    add_prob = np.random.rand()
                    if add_prob <= add_probability:
                        add_word = self._generate_additional_token(
                            text, local_epsilon, lm_model, raw_model, tokenizer, **kwargs
                        )
                        if add_word:
                            add_pos = adjusted_start + len(private_token)
                            result = result[:add_pos] + " " + add_word + result[add_pos:]
                            offset_adjust += len(" " + add_word)
                            added += 1
                else:
                    
                    result = result[:adjusted_start] + result[adjusted_end:]
                    offset_adjust -= len(span.text)
                    deleted += 1
        
        result = PrivacyResult(
            original_text=text,
            private_text=result,
            perturbed_tokens=perturbed,
            total_tokens=total,
            added_tokens=added,
            deleted_tokens=deleted
        )

        result.metadata = result.metadata or {}
        result.metadata["mechanism"] = "dpmlm"
        result.metadata["scoring_strategy"] = self.scoring_strategy.name

        token_allocations = [alloc.to_metadata(idx + 1) for idx, alloc in enumerate(allocations)]
        result.metadata["token_allocations"] = token_allocations

        top_k = min(self.settings.summary_top_k, len(token_allocations))
        if top_k > 0:
            result.metadata["token_allocations_summary"] = token_allocations[:top_k]

        return result
    
    def _privatize_single_token(self, text: str, token: str, epsilon: float, 
                               lm_model, raw_model, tokenizer, **kwargs) -> str:
        """Privatize a single token using the model."""
        # Get hyperparameters from kwargs with defaults
        use_temperature = kwargs.get('use_temperature', kwargs.get('TEMP', self.config.use_temperature))
        k_candidates = kwargs.get('k_candidates', kwargs.get('K', self.config.k_candidates))
        
        
        
        
        masked_text = text.replace(token, tokenizer.mask_token, 1)


        input_ids = tokenizer.encode(masked_text, add_special_tokens=True)

        try:
            mask_pos = input_ids.index(tokenizer.mask_token_id)
        except ValueError:
            return token

        max_len = self.config.model_config.max_sequence_length
        aggregated_logits = None
        mask_count = 0

        try:
            chunked_inputs = self.core._chunk_sequence_if_needed(input_ids, [mask_pos])

            for chunk_ids, chunk_mask_positions in chunked_inputs:
                if not chunk_mask_positions:
                    continue

                if not chunk_ids:
                    continue

                model_input = torch.tensor(chunk_ids).reshape(1, -1).to(self.device)

                with torch.no_grad():
                    output = lm_model(model_input)

                logits = output[0].squeeze().detach().cpu().numpy()

                for chunk_mask_pos in chunk_mask_positions:
                    if chunk_mask_pos < 0 or chunk_mask_pos >= logits.shape[0]:
                        continue
                    mask_logits_chunk = logits[chunk_mask_pos]
                    aggregated_logits = (
                        mask_logits_chunk.copy()
                        if aggregated_logits is None
                        else aggregated_logits + mask_logits_chunk
                    )
                    mask_count += 1
        except Exception as exc:  # pragma: no cover - defensive safety net
            logger.debug("Chunked inference failed; falling back to truncation: %s", exc)
            chunked_inputs = []

        if aggregated_logits is None or mask_count == 0:
            truncated_ids = input_ids[:max_len]
            try:
                fallback_mask_pos = truncated_ids.index(tokenizer.mask_token_id)
            except ValueError:
                return token

            model_input = torch.tensor(truncated_ids).reshape(1, -1).to(self.device)

            with torch.no_grad():
                output = lm_model(model_input)

            logits = output[0].squeeze().detach().cpu().numpy()
            mask_logits = logits[fallback_mask_pos]
        else:
            mask_logits = aggregated_logits / mask_count

        if use_temperature:
            
            temperature = self.core._calculate_temperature(epsilon)
            mask_logits = np.clip(mask_logits, self.config.clip_min, self.config.clip_max)
            mask_logits = mask_logits / temperature
            
            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
            scores = scores / scores.sum()
            
            
            chosen_idx = np.random.choice(len(mask_logits), p=scores.numpy())
            return tokenizer.decode(chosen_idx).strip()
        else:
            
            top_tokens = torch.topk(torch.from_numpy(mask_logits), k=k_candidates, dim=0)[1]
            return tokenizer.decode(top_tokens[0].item()).strip()
    
    def _generate_additional_token(self, text: str, epsilon: float, 
                                  lm_model, raw_model, tokenizer, **kwargs) -> str:
        """Generate an additional token for insertion."""
        # Get hyperparameters from kwargs with defaults
        use_temperature = kwargs.get('use_temperature', kwargs.get('TEMP', self.config.use_temperature))
        k_candidates = kwargs.get('k_candidates', kwargs.get('K', self.config.k_candidates))
        
        masked_text = text + " " + tokenizer.mask_token

        input_ids = tokenizer.encode(masked_text, add_special_tokens=True)

        try:
            mask_pos = input_ids.index(tokenizer.mask_token_id)
        except ValueError:
            return ""

        max_len = self.config.model_config.max_sequence_length
        aggregated_logits = None
        mask_count = 0

        try:
            chunked_inputs = self.core._chunk_sequence_if_needed(input_ids, [mask_pos])

            for chunk_ids, chunk_mask_positions in chunked_inputs:
                if not chunk_mask_positions:
                    continue

                if not chunk_ids:
                    continue

                model_input = torch.tensor(chunk_ids).reshape(1, -1).to(self.device)

                with torch.no_grad():
                    output = lm_model(model_input)

                logits = output[0].squeeze().detach().cpu().numpy()

                for chunk_mask_pos in chunk_mask_positions:
                    if chunk_mask_pos < 0 or chunk_mask_pos >= logits.shape[0]:
                        continue
                    mask_logits_chunk = logits[chunk_mask_pos]
                    aggregated_logits = (
                        mask_logits_chunk.copy()
                        if aggregated_logits is None
                        else aggregated_logits + mask_logits_chunk
                    )
                    mask_count += 1
        except Exception as exc:  # pragma: no cover - defensive safety net
            logger.debug("Chunked inference failed; falling back to truncation: %s", exc)
            chunked_inputs = []

        if aggregated_logits is None or mask_count == 0:
            truncated_ids = input_ids[-max_len:]
            try:
                fallback_mask_pos = truncated_ids.index(tokenizer.mask_token_id)
            except ValueError:
                return ""

            model_input = torch.tensor(truncated_ids).reshape(1, -1).to(self.device)

            with torch.no_grad():
                output = lm_model(model_input)

            logits = output[0].squeeze().detach().cpu().numpy()
            mask_logits = logits[fallback_mask_pos]
        else:
            mask_logits = aggregated_logits / mask_count

        if use_temperature:
            temperature = self.core._calculate_temperature(epsilon)
            mask_logits = np.clip(mask_logits, self.config.clip_min, self.config.clip_max)
            mask_logits = mask_logits / temperature
            
            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
            scores = scores / scores.sum()
            
            chosen_idx = np.random.choice(len(mask_logits), p=scores.numpy())
            return tokenizer.decode(chosen_idx).strip()
        else:
            top_tokens = torch.topk(torch.from_numpy(mask_logits), k=k_candidates, dim=0)[1]
            return tokenizer.decode(top_tokens[0].item()).strip()

    # ------------------------------------------------------------------
    # Strategy factory helpers
    # ------------------------------------------------------------------
    def _build_selection_strategy(self) -> TokenSelectionStrategy:
        if self.config.process_pii_only:
            return PIITokenSelection(annotator=self.settings.annotator, threshold=self.settings.pii_threshold)
        return AllTokensSelection()

    def _build_scoring_strategy(self) -> ScoringStrategy:
        mode = (self.settings.explainability_mode or "uniform").lower()
        if mode == "greedy":
            return GreedyScoring()
        if mode == "shap":
            return ShapScoring()
        return UniformScoring()

    # ------------------------------------------------------------------
    # Weight computation and allocations
    # ------------------------------------------------------------------
    def _compute_ranking_weights(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores

        clipped = scores.astype(float)
        if self.settings.clip_contribution is not None:
            clipped = np.clip(clipped, 0.0, self.settings.clip_contribution)

        positive = np.maximum(clipped, self.settings.min_weight)
        total = float(positive.sum())
        if total <= 0.0:
            ranking = np.full_like(positive, 1.0 / positive.size)
        else:
            ranking = positive / total

        if self.settings.top_k is not None and self.settings.top_k < ranking.size:
            top_indices = np.argsort(ranking)[::-1][: self.settings.top_k]
            mask = np.zeros_like(ranking)
            mask[top_indices] = 1.0
            masked = ranking * mask
            masked_sum = masked.sum()
            ranking = masked / masked_sum if masked_sum > 0 else np.full_like(ranking, 1.0 / ranking.size)

        return ranking

    def _compute_privacy_epsilons(self, ranking_weights: np.ndarray, epsilon: float) -> np.ndarray:
        if ranking_weights.size == 0:
            return ranking_weights

        denom = np.maximum(ranking_weights, self.settings.min_weight)
        scale = denom * max(len(ranking_weights), 1)
        eps_values = epsilon / scale
        max_eps = epsilon * max(len(ranking_weights), 1)
        eps_values = np.clip(eps_values, self.settings.min_weight, max_eps)
        return eps_values

    def _build_allocations(
        self,
        spans: List[TokenSpan],
        scores: np.ndarray,
        ranking_weights: np.ndarray,
        epsilon: float,
    ) -> List[TokenRiskAllocation]:
        if not spans:
            return []

        if ranking_weights.size != len(spans):
            ranking_weights = np.full(len(spans), 1.0 / max(len(spans), 1))

        epsilon_values = self._compute_privacy_epsilons(ranking_weights, epsilon)

        return [
            TokenRiskAllocation(
                span=span,
                score=float(score),
                weight=float(weight),
                epsilon=float(token_eps),
            )
            for span, score, weight, token_eps in zip(spans, scores, ranking_weights, epsilon_values)
        ]

    # ------------------------------------------------------------------
    # Helpers bridging to scoring strategies
    # ------------------------------------------------------------------
    def _resolve_label_index(self, pipeline: Any, text: str) -> int:
        label_setting = self.settings.explainability_label
        model = getattr(pipeline, "model", None)
        config = getattr(model, "config", None)
        label2id = getattr(config, "label2id", {}) if config else {}
        id2label = getattr(config, "id2label", {}) if config else {}

        if isinstance(label_setting, int):
            return label_setting

        if isinstance(label_setting, str):
            key = label_setting.lower()
            if label2id:
                lookup = {k.lower(): v for k, v in label2id.items()}
                return lookup.get(key, 0)
            for idx, name in id2label.items():
                if name.lower() == key:
                    return int(idx)
            return 0

        prediction = pipeline(text, truncation=True)
        if isinstance(prediction, list) and prediction:
            top = prediction[0]
            label_name = top.get("label") if isinstance(top, dict) else None
            if label_name and label2id:
                lookup = {k.lower(): v for k, v in label2id.items()}
                return lookup.get(label_name.lower(), 0)
            if label_name and id2label:
                for idx, name in id2label.items():
                    if name.lower() == label_name.lower():
                        return int(idx)

        if config and hasattr(config, "num_labels"):
            return int(getattr(config, "num_labels") // 2)
        return 0

    def _label_probability(self, pipeline: Any, text: str, label_index: int) -> Optional[float]:
        try:
            predictions = pipeline(text, return_all_scores=True, truncation=True)
        except TypeError:
            predictions = pipeline(text, truncation=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("Risk pipeline inference failed: %s", exc)
            return None

        if not predictions:
            return None

        scores = predictions[0]
        model = getattr(pipeline, "model", None)
        config = getattr(model, "config", None)
        id2label = getattr(config, "id2label", {}) if config else {}

        if isinstance(scores, list) and scores and isinstance(scores[0], dict):
            if id2label:
                target_label = id2label.get(label_index) or id2label.get(str(label_index))
                if target_label is not None:
                    for entry in scores:
                        if entry.get("label") == target_label:
                            return float(entry.get("score", 0.0))
            if 0 <= label_index < len(scores):
                return float(scores[label_index].get("score", 0.0))
            return float(scores[0].get("score", 0.0))

        if isinstance(scores, dict):
            return float(scores.get(label_index, 0.0))

        if isinstance(scores, list) and scores and np.isscalar(scores[0]):
            if 0 <= label_index < len(scores):
                return float(scores[label_index])

        return None

    def _mask_span(self, text: str, span: TokenSpan, mask_text: str) -> str:
        return text[: span.start] + mask_text + text[span.end :]

    def _terms_to_tokens(self, text: str, spans: List[TokenSpan], tokenizer: Any) -> List[List[int]]:
        if not spans:
            return []

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping", [])
        terms_to_tokens: List[List[int]] = []
        last_idx = 0

        for span in spans:
            start, end = span.start, span.end
            token_indices: List[int] = []
            for token_idx in range(last_idx, len(offsets)):
                token_start, token_end = offsets[token_idx]
                if token_end <= start:
                    continue
                if token_start >= end:
                    last_idx = token_idx
                    break
                token_indices.append(token_idx)
            else:
                last_idx = len(offsets)
            terms_to_tokens.append(token_indices)

        return terms_to_tokens

    def _ensure_shap_explainer(self, pipeline: Any):
        if self._shap_explainer is not None:
            return self._shap_explainer

        try:
            import shap  # type: ignore
        except ImportError:
            logger.warning("SHAP library not available; scoring falls back to uniform weights.")
            return None

        try:
            self._shap_explainer = shap.Explainer(pipeline, silent=self.settings.shap_silent)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to initialise SHAP explainer: %s", exc)
            self._shap_explainer = None
        return self._shap_explainer
