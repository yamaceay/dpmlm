"""
Configuration classes for differential privacy mechanisms.

This module provides type-safe configuration classes with validation
and default values for all differential privacy mechanisms.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    """Base configuration class with common parameters."""
    epsilon: float = 1.0
    device: str = "auto"  
    seed: Optional[int] = 42
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {self.epsilon}")
        if self.device not in ["auto", "cuda", "cpu", "mps"]:
            raise ValueError(f"Invalid device: {self.device}")


@dataclass
class ModelConfig:
    """Configuration for transformer models."""
    model_name: str = "roberta-base"
    max_sequence_length: int = 512
    truncation: bool = True
    padding: bool = True
    cache_dir: Optional[str] = None
    
    def validate(self) -> None:
        """Validate model configuration."""
        if self.max_sequence_length <= 0:
            raise ValueError(f"Max sequence length must be positive, got {self.max_sequence_length}")


@dataclass
class DPMLMConfig(BaseConfig):
    """Configuration for DPMLM mechanism."""
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    
    alpha: float = 0.003
    clip_min: float = -3.2093127
    clip_max: float = 16.304797887802124
    
    
    use_treebank_tokenizer: bool = True
    preserve_case: bool = True
    
    
    k_candidates: int = 5
    filter_words: bool = True
    use_pos_tags: bool = True
    concat_sentences: bool = True
    english_only: bool = True
    use_temperature: bool = True
    
    
    process_pii_only: bool = True
    annotator_config: Optional[Dict[str, Any]] = None
    
    
    add_probability: float = 0.15
    delete_probability: float = 0.05

    def __post_init__(self):
        """Post-initialization processing."""
        
        if isinstance(self.model_config, dict):
            self.model_config = ModelConfig(**self.model_config)
        elif self.model_config is None:
            self.model_config = ModelConfig()
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate DPMLM configuration."""
        super().validate()
        self.model_config.validate()
        
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}")
        if self.clip_min >= self.clip_max:
            raise ValueError(f"clip_min ({self.clip_min}) must be less than clip_max ({self.clip_max})")
        if self.k_candidates <= 0:
            raise ValueError(f"k_candidates must be positive, got {self.k_candidates}")
        if not 0 <= self.add_probability <= 1:
            raise ValueError(f"add_probability must be in [0,1], got {self.add_probability}")
        if not 0 <= self.delete_probability <= 1:
            raise ValueError(f"delete_probability must be in [0,1], got {self.delete_probability}")

    @property
    def sensitivity(self) -> float:
        """Calculate sensitivity from clip values."""
        return abs(self.clip_max - self.clip_min)


@dataclass  
class DPPromptConfig(BaseConfig):
    """Configuration for DPPrompt mechanism."""
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_name="google/flan-t5-base",
        max_sequence_length=512
    ))
    
    
    min_logit: float = -19.22705113016047
    max_logit: float = 7.48324937989716
    
    
    max_new_tokens: Optional[int] = None  
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = True
    
    
    prompt_template: str = "Document : {}\nParaphrase of the document :"
    
    def __post_init__(self):
        """Post-initialization processing."""
        
        if isinstance(self.model_config, dict):
            self.model_config = ModelConfig(**self.model_config)
        elif self.model_config is None:
            self.model_config = ModelConfig(
                model_name="google/flan-t5-base",
                max_sequence_length=512
            )
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate DPPrompt configuration."""
        super().validate()
        self.model_config.validate()
        
        if self.min_logit >= self.max_logit:
            raise ValueError(f"min_logit ({self.min_logit}) must be less than max_logit ({self.max_logit})")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if not 0 <= self.top_p <= 1:
            raise ValueError(f"top_p must be in [0,1], got {self.top_p}")

    @property
    def sensitivity(self) -> float:
        """Calculate sensitivity from logit range."""
        return abs(self.max_logit - self.min_logit)


@dataclass
class DPParaphraseConfig(BaseConfig):
    """Configuration for DPParaphrase mechanism."""
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_name="models/gpt2-paraphraser",
        max_sequence_length=512
    ))
    
    
    min_logit: float = -96.85249956065758
    max_logit: float = -8.747697966442914
    
    
    max_new_tokens: Optional[int] = None  
    
    
    prompt_suffix: str = " >>>>> "
    output_cleanup_chars: List[str] = field(default_factory=lambda: [">", "\xa0"])
    
    def __post_init__(self):
        """Post-initialization processing."""
        
        if isinstance(self.model_config, dict):
            self.model_config = ModelConfig(**self.model_config)
        elif self.model_config is None:
            self.model_config = ModelConfig(
                model_name="models/gpt2-paraphraser",
                max_sequence_length=512
            )
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate DPParaphrase configuration."""
        super().validate()
        self.model_config.validate()
        
        if self.min_logit >= self.max_logit:
            raise ValueError(f"min_logit ({self.min_logit}) must be less than max_logit ({self.max_logit})")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")

    @property
    def sensitivity(self) -> float:
        """Calculate sensitivity from logit range."""
        return abs(self.max_logit - self.min_logit)


@dataclass
class DPBartConfig(BaseConfig):
    """Configuration for DPBart mechanism."""
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_name="facebook/bart-base",
        max_sequence_length=512
    ))
    
    
    sigma: float = 0.2
    num_sigmas: float = 0.5
    delta: float = 1e-5
    
    
    noise_method: str = "gaussian"  
    
    
    analytic_tolerance: float = 1e-12
    precision_dps: Dict[str, int] = field(default_factory=lambda: {
        "1000": 500,
        "2500": 1100,
        "5000": 2200
    })
    
    def __post_init__(self):
        """Post-initialization processing."""
        
        if isinstance(self.model_config, dict):
            self.model_config = ModelConfig(**self.model_config)
        elif self.model_config is None:
            self.model_config = ModelConfig(
                model_name="facebook/bart-base",
                max_sequence_length=512
            )
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate DPBart configuration."""
        super().validate()
        self.model_config.validate()
        
        if self.sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {self.sigma}")
        if self.num_sigmas <= 0:
            raise ValueError(f"num_sigmas must be positive, got {self.num_sigmas}")
        if self.delta <= 0:
            raise ValueError(f"Delta must be positive, got {self.delta}")
        if self.noise_method not in ["gaussian", "laplace", "analytic_gaussian"]:
            raise ValueError(f"Invalid noise method: {self.noise_method}")
        if self.analytic_tolerance <= 0:
            raise ValueError(f"analytic_tolerance must be positive, got {self.analytic_tolerance}")

    @property
    def c_min(self) -> float:
        """Calculate minimum clipping value."""
        return -self.sigma
    
    @property  
    def c_max(self) -> float:
        """Calculate maximum clipping value."""
        return self.num_sigmas * self.sigma


@dataclass
class AnnotatorConfig:
    """Configuration for PII annotators."""
    model_path: str
    unique_labels: List[str] = field(default_factory=lambda: [
        'CODE', 'DEM', 'ORG', 'QUANTITY', 'LOC', 'DATETIME', 'MISC', 'PERSON'
    ])
    output_dir: str = "pii/outputs"
    
    def validate(self) -> None:
        """Validate annotator configuration."""
        if not Path(self.model_path).exists():
            logger.warning("Model path does not exist: %s", self.model_path)
        if not self.unique_labels:
            raise ValueError("unique_labels cannot be empty")



def create_dpmlm_config(**kwargs) -> DPMLMConfig:
    """Create DPMLM configuration with custom parameters."""
    return DPMLMConfig(**kwargs)

def create_dpprompt_config(**kwargs) -> DPPromptConfig:
    """Create DPPrompt configuration with custom parameters."""
    return DPPromptConfig(**kwargs)

def create_dpparaphrase_config(**kwargs) -> DPParaphraseConfig:
    """Create DPParaphrase configuration with custom parameters."""
    return DPParaphraseConfig(**kwargs)

def create_dpbart_config(**kwargs) -> DPBartConfig:
    """Create DPBart configuration with custom parameters."""
    return DPBartConfig(**kwargs)

def create_annotator_config(**kwargs) -> AnnotatorConfig:
    """Create annotator configuration with custom parameters."""
    return AnnotatorConfig(**kwargs)
