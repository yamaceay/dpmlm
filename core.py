"""
Core DPMLM components with high-level, type-safe interfaces.

This module provides the main DPMLM mechanism implementation following
the same design patterns as the PII module.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import string
import numpy as np
import torch
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from collections import Counter

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
            # Fallback to simple word tokenization
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
    
    def _chunk_sequence_if_needed(self, input_ids: List[int], mask_positions: List[int]) -> Tuple[List[int], List[int]]:
        """Chunk sequence if it exceeds max length."""
        max_len = self.config.model_config.max_sequence_length
        
        if len(input_ids) <= max_len:
            return input_ids, mask_positions
            
        # Find optimal chunk around mask positions
        if mask_positions:
            mask_pos = mask_positions[0]
            available_tokens = max_len - 2  # Account for special tokens
            context_per_side = available_tokens // 2
            
            start_pos = max(0, mask_pos - context_per_side)
            end_pos = min(len(input_ids), mask_pos + context_per_side + 1)
            
            if start_pos == 0:
                end_pos = min(len(input_ids), max_len)
            elif end_pos == len(input_ids):
                start_pos = max(0, len(input_ids) - max_len)
            
            # Adjust mask positions
            adjusted_mask_positions = [
                pos - start_pos for pos in mask_positions 
                if start_pos <= pos < end_pos
            ]
            
            return input_ids[start_pos:end_pos], adjusted_mask_positions
        
        # If no mask positions, just truncate
        return input_ids[:max_len], []


class DPMLMPrivatizer(DPMechanism):
    """High-level DPMLM privatizer implementing the DPMechanism interface."""
    
    def __init__(self, config: DPMLMConfig, annotator=None):
        self.config = config
        self.core = DPMLMCore(config)
        self.annotator = annotator
        self._setup_resources()
        
    def _setup_resources(self) -> None:
        """Setup model resources."""
        self.device = self.core._get_device()
        logger.info("DPMLM privatizer initialized on device: %s", self.device)
        
    def validate_epsilon(self, epsilon: float) -> bool:
        """Validate epsilon parameter."""
        return epsilon > 0
        
    def _mark_critical_tokens(self, spans: List[TokenSpan], text: str) -> List[TokenSpan]:
        """Mark critical (PII) tokens if annotator is available."""
        if not self.annotator or not self.config.process_pii_only:
            # Mark all tokens as critical if no annotator
            for span in spans:
                span.is_critical = True
            return spans
            
        # Use annotator to identify PII tokens
        predictions = self.annotator.predict(text)
        unique_labels = getattr(self.annotator.labels, 'unique_labels', [])
        
        for prediction in predictions:
            if prediction.get('entity_group') in unique_labels:
                start, end = prediction['start'], prediction['end']
                for span in spans:
                    if (span.start <= start < span.end or 
                        span.start < end <= span.end):
                        span.is_critical = True
                        span.entity_type = prediction.get('entity_group')
        
        return spans
    
    def privatize(self, text: str, epsilon: float, **kwargs) -> PrivacyResult:
        """Apply differential privacy to text."""
        if not self.validate_epsilon(epsilon):
            raise ValueError(f"Invalid epsilon: {epsilon}")
            
        method = kwargs.get('method', 'patch')
        use_plus = kwargs.get('plus', False)
        
        if method == 'patch':
            if use_plus:
                return self._privatize_patch_plus(text, epsilon, **kwargs)
            else:
                return self._privatize_patch(text, epsilon, **kwargs)
        else:
            if use_plus:
                return self._privatize_plus(text, epsilon, **kwargs)
            else:
                return self._privatize_basic(text, epsilon, **kwargs)
    
    def _privatize_patch(self, text: str, epsilon: float, **kwargs) -> PrivacyResult:
        """Apply patch-based privatization."""
        spans = self.core.tokenizer_wrapper.tokenize_with_spans(text)
        spans = self._mark_critical_tokens(spans, text)
        
        perturbed = 0
        total = 0
        result = text
        offset_adjust = 0
        
        with DPMLMModelManager(self.config.model_config, str(self.device)) as (models, tokenizer):
            lm_model, raw_model = models
            
            for span in spans:
                if not span.is_critical:
                    total += 1
                    continue
                    
                if span.text in string.punctuation:
                    total += 1
                    continue
                
                # Apply privatization to this token
                adjusted_start = span.start + offset_adjust
                adjusted_end = span.end + offset_adjust
                
                private_token = self._privatize_single_token(
                    text, span.text, epsilon, lm_model, raw_model, tokenizer
                )
                
                # Handle capitalization
                if span.text and span.text[0].isupper():
                    private_token = private_token.capitalize() if private_token else span.text
                elif span.text and span.text[0].islower():
                    private_token = private_token.lower() if private_token else span.text
                
                # Apply replacement
                result = result[:adjusted_start] + private_token + result[adjusted_end:]
                offset_adjust += len(private_token) - len(span.text)
                
                if private_token != span.text:
                    perturbed += 1
                total += 1
        
        return PrivacyResult(
            original_text=text,
            private_text=result,
            perturbed_tokens=perturbed,
            total_tokens=total
        )
    
    def _privatize_patch_plus(self, text: str, epsilon: float, **kwargs) -> PrivacyResult:
        """Apply patch-based privatization with addition/deletion."""
        spans = self.core.tokenizer_wrapper.tokenize_with_spans(text)
        spans = self._mark_critical_tokens(spans, text)
        
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
                
                adjusted_start = span.start + offset_adjust
                adjusted_end = span.end + offset_adjust
                
                # Deletion logic
                if i == len(spans) - 1:
                    delete_prob = 1.0  # Never delete last token
                else:
                    delete_prob = np.random.rand()
                
                if delete_prob >= self.config.delete_probability:
                    # Replace token
                    private_token = self._privatize_single_token(
                        text, span.text, epsilon, lm_model, raw_model, tokenizer
                    )
                    
                    # Handle capitalization
                    if span.text and span.text[0].isupper():
                        private_token = private_token.capitalize() if private_token else span.text
                    elif span.text and span.text[0].islower():
                        private_token = private_token.lower() if private_token else span.text
                    
                    # Apply replacement
                    result = result[:adjusted_start] + private_token + result[adjusted_end:]
                    offset_adjust += len(private_token) - len(span.text)
                    
                    if private_token != span.text:
                        perturbed += 1
                    total += 1
                    
                    # Addition logic
                    add_prob = np.random.rand()
                    if add_prob <= self.config.add_probability:
                        add_word = self._generate_additional_token(
                            text, epsilon, lm_model, raw_model, tokenizer
                        )
                        if add_word:
                            add_pos = adjusted_start + len(private_token)
                            result = result[:add_pos] + " " + add_word + result[add_pos:]
                            offset_adjust += len(" " + add_word)
                            added += 1
                else:
                    # Delete token
                    result = result[:adjusted_start] + result[adjusted_end:]
                    offset_adjust -= len(span.text)
                    deleted += 1
        
        return PrivacyResult(
            original_text=text,
            private_text=result,
            perturbed_tokens=perturbed,
            total_tokens=total,
            added_tokens=added,
            deleted_tokens=deleted
        )
    
    def _privatize_basic(self, text: str, epsilon: float, **kwargs) -> PrivacyResult:
        """Apply basic (non-patch) privatization."""
        # Implementation for basic method would go here
        # For now, delegate to patch method
        return self._privatize_patch(text, epsilon, **kwargs)
    
    def _privatize_plus(self, text: str, epsilon: float, **kwargs) -> PrivacyResult:
        """Apply basic plus privatization."""
        # Implementation for basic plus method would go here
        # For now, delegate to patch plus method
        return self._privatize_patch_plus(text, epsilon, **kwargs)
    
    def _privatize_single_token(self, text: str, token: str, epsilon: float, 
                               lm_model, raw_model, tokenizer) -> str:
        """Privatize a single token using the model."""
        # This is a simplified version - full implementation would include
        # all the logic from the original privatize_patch method
        
        # Create masked sentence
        masked_text = text.replace(token, tokenizer.mask_token, 1)
        
        # Tokenize and handle length
        input_ids = tokenizer.encode(masked_text, add_special_tokens=True)
        
        if len(input_ids) > self.config.model_config.max_sequence_length:
            try:
                mask_pos = input_ids.index(tokenizer.mask_token_id)
                input_ids, _ = self.core._chunk_sequence_if_needed(input_ids, [mask_pos])
                mask_pos = input_ids.index(tokenizer.mask_token_id) if tokenizer.mask_token_id in input_ids else -1
            except ValueError:
                return token  # Fallback if mask token not found
        else:
            try:
                mask_pos = input_ids.index(tokenizer.mask_token_id)
            except ValueError:
                return token
        
        if mask_pos == -1:
            return token
        
        # Get model predictions
        model_input = torch.tensor(input_ids).reshape(1, -1).to(self.device)
        
        with torch.no_grad():
            output = lm_model(model_input)
        
        logits = output[0].squeeze().detach().cpu().numpy()
        mask_logits = logits[mask_pos]
        
        if self.config.use_temperature:
            # Apply temperature scaling
            temperature = self.core._calculate_temperature(epsilon)
            mask_logits = np.clip(mask_logits, self.config.clip_min, self.config.clip_max)
            mask_logits = mask_logits / temperature
            
            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
            scores = scores / scores.sum()
            
            # Sample from distribution
            chosen_idx = np.random.choice(len(mask_logits), p=scores.numpy())
            return tokenizer.decode(chosen_idx).strip()
        else:
            # Use top-k selection
            top_tokens = torch.topk(torch.from_numpy(mask_logits), k=self.config.k_candidates, dim=0)[1]
            return tokenizer.decode(top_tokens[0].item()).strip()
    
    def _generate_additional_token(self, text: str, epsilon: float, 
                                  lm_model, raw_model, tokenizer) -> str:
        """Generate an additional token for insertion."""
        # Use MASK token to generate new content
        masked_text = text + " " + tokenizer.mask_token
        
        input_ids = tokenizer.encode(masked_text, add_special_tokens=True)
        
        if len(input_ids) > self.config.model_config.max_sequence_length:
            input_ids = input_ids[-self.config.model_config.max_sequence_length:]
        
        try:
            mask_pos = input_ids.index(tokenizer.mask_token_id)
        except ValueError:
            return ""
        
        model_input = torch.tensor(input_ids).reshape(1, -1).to(self.device)
        
        with torch.no_grad():
            output = lm_model(model_input)
        
        logits = output[0].squeeze().detach().cpu().numpy()
        mask_logits = logits[mask_pos]
        
        if self.config.use_temperature:
            temperature = self.core._calculate_temperature(epsilon)
            mask_logits = np.clip(mask_logits, self.config.clip_min, self.config.clip_max)
            mask_logits = mask_logits / temperature
            
            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
            scores = scores / scores.sum()
            
            chosen_idx = np.random.choice(len(mask_logits), p=scores.numpy())
            return tokenizer.decode(chosen_idx).strip()
        else:
            top_tokens = torch.topk(torch.from_numpy(mask_logits), k=self.config.k_candidates, dim=0)[1]
            return tokenizer.decode(top_tokens[0].item()).strip()


# Factory function for easy instantiation
def create_dpmlm_privatizer(config: Optional[DPMLMConfig] = None, annotator=None) -> DPMLMPrivatizer:
    """Create a DPMLM privatizer with optional configuration."""
    if config is None:
        config = DPMLMConfig()
    return DPMLMPrivatizer(config, annotator)
