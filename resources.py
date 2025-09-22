"""
Resource management system for differential privacy models.

This module provides context managers and resource management utilities
similar to the PII module's TorchTokenClassifier pattern.
"""

import logging
from typing import Tuple, Any, Optional, Dict
from contextlib import contextmanager
import torch
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, AutoModel,
    AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    BartTokenizer, BartModel, BartForConditionalGeneration
)

from dpmlm.config import ModelConfig

logger = logging.getLogger(__name__)


def get_optimal_device() -> torch.device:
    """Get the optimal device for model inference."""
    if torch.backends.mps.is_available():
        device_str = "mps"
        logger.info("Using MPS device (Apple Silicon)")
        if not torch.backends.mps.is_built():
            logger.warning("MPS not built, falling back to CPU")
            device_str = "cpu"
    elif torch.cuda.is_available():
        device_str = "cuda"
        logger.info("Using CUDA device")
    else:
        device_str = "cpu"
        logger.info("Using CPU device")
    
    return torch.device(device_str)


class ModelManager:
    """Base context manager for model and tokenizer resource management."""
    
    def __init__(self, config: ModelConfig, device: Optional[str] = None):
        self.config = config
        self.device = get_optimal_device() if device == "auto" or device is None else torch.device(device)
        self.model = None
        self.tokenizer = None
        
    def _load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load model and tokenizer. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _load_model_and_tokenizer")
    
    def __enter__(self) -> Tuple[Any, Any]:
        """Enter context and load resources."""
        logger.info("Loading model: %s", self.config.model_name)
        
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        if hasattr(self.model, 'parameters'):
            num_params = sum(p.numel() for p in self.model.parameters())
            logger.info("Model loaded with %.2f M parameters on %s", num_params / 1e6, self.device)
        
        return self.model, self.tokenizer
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and cleanup resources."""
        if self.model:
            if isinstance(self.model, tuple):
                for m in self.model:
                    m.cpu()
            else:
                self.model.cpu()
            del self.model
            self.model = None
            
        if self.tokenizer:
            del self.tokenizer  
            self.tokenizer = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Resources cleaned up")


class DPMLMModelManager(ModelManager):
    """Model manager for DPMLM models (RoBERTa-based)."""
    
    def _load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load RoBERTa model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        
        # Load both masked LM and raw models
        lm_model = AutoModelForMaskedLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        ).to(self.device)
        
        raw_model = AutoModel.from_pretrained(
            self.config.model_name,
            output_hidden_states=True,
            output_attentions=True,
            cache_dir=self.config.cache_dir
        ).to(self.device)
        
        # Return both models as a tuple
        return (lm_model, raw_model), tokenizer


class DPPromptModelManager(ModelManager):
    """Model manager for DPPrompt models (T5-based)."""
    
    def _load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load T5 model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        ).to(self.device)
        
        return model, tokenizer


class DPParaphraseModelManager(ModelManager):
    """Model manager for DPParaphrase models (GPT2-based)."""
    
    def _load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load GPT2 model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            pad_token_id=tokenizer.eos_token_id,
            cache_dir=self.config.cache_dir
        ).to(self.device)
        
        return model, tokenizer


class DPBartModelManager(ModelManager):
    """Model manager for DPBart models (BART-based)."""
    
    def _load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load BART model and tokenizer."""
        tokenizer = BartTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        
        # Load both encoder and decoder models
        encoder_model = BartModel.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        ).to(self.device)
        
        decoder_model = BartForConditionalGeneration.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        ).to(self.device)
        
        return (encoder_model, decoder_model), tokenizer


@contextmanager
def dpmlm_models(config: ModelConfig, device: Optional[str] = None):
    """Context manager for DPMLM models."""
    manager = DPMLMModelManager(config, device)
    with manager as (models, tokenizer):
        yield models, tokenizer


@contextmanager  
def dpprompt_model(config: ModelConfig, device: Optional[str] = None):
    """Context manager for DPPrompt model."""
    manager = DPPromptModelManager(config, device)
    with manager as (model, tokenizer):
        yield model, tokenizer


@contextmanager
def dpparaphrase_model(config: ModelConfig, device: Optional[str] = None):
    """Context manager for DPParaphrase model."""
    manager = DPParaphraseModelManager(config, device)
    with manager as (model, tokenizer):
        yield model, tokenizer


@contextmanager
def dpbart_models(config: ModelConfig, device: Optional[str] = None):
    """Context manager for DPBart models."""
    manager = DPBartModelManager(config, device)
    with manager as (models, tokenizer):
        yield models, tokenizer


class ResourcePool:
    """Pool for managing multiple model instances."""
    
    def __init__(self):
        self._managers: Dict[str, ModelManager] = {}
        
    def get_manager(self, key: str, manager_class: type, config: ModelConfig, device: Optional[str] = None) -> ModelManager:
        """Get or create a model manager."""
        if key not in self._managers:
            self._managers[key] = manager_class(config, device)
        return self._managers[key]
    
    def cleanup(self) -> None:
        """Cleanup all managed resources."""
        for manager in self._managers.values():
            if hasattr(manager, '_cleanup'):
                manager._cleanup()
        self._managers.clear()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()
