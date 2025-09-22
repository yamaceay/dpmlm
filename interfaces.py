"""
Base interfaces and protocols for differential privacy text processing.

This module defines abstract base classes and protocols that establish
type-safe, plug-and-play interfaces for different components of the
differential privacy text processing pipeline.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, List, Tuple, Any, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedModel


@dataclass
class PrivacyResult:
    """Result container for privacy-preserving text transformations."""
    original_text: str
    private_text: str
    perturbed_tokens: int
    total_tokens: int
    added_tokens: int = 0
    deleted_tokens: int = 0
    metadata: Optional[Dict[str, Any]] = None

    @property
    def perturbation_rate(self) -> float:
        """Calculate the rate of token perturbation."""
        return self.perturbed_tokens / max(self.total_tokens, 1)


@dataclass
class TokenSpan:
    """Represents a token span with position information."""
    text: str
    start: int
    end: int
    is_critical: bool = False
    entity_type: Optional[str] = None


class DPTokenizer(Protocol):
    """Protocol for differential privacy tokenizers."""
    
    def tokenize_with_spans(self, text: str) -> List[TokenSpan]:
        """Tokenize text and return token spans with position information."""
        raise NotImplementedError
    
    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        raise NotImplementedError


class PIIAnnotator(Protocol):
    """Protocol for PII (Personally Identifiable Information) annotators."""
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Predict PII entities in text."""
        raise NotImplementedError
    
    @property
    def unique_labels(self) -> List[str]:
        """Get unique entity labels supported by this annotator."""
        raise NotImplementedError


class DPModel(Protocol):
    """Protocol for differential privacy models."""
    
    def privatize_token(
        self, 
        sentence: str, 
        target: Union[str, List[str]], 
        epsilon: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Privatize specific tokens in a sentence."""
        raise NotImplementedError
    
    @property
    def max_sequence_length(self) -> int:
        """Maximum sequence length supported by the model."""
        raise NotImplementedError


class DPMechanism(ABC):
    """Abstract base class for differential privacy mechanisms."""
    
    @abstractmethod
    def privatize(self, text: str, epsilon: float, **kwargs) -> PrivacyResult:
        """Apply differential privacy to text."""
        raise NotImplementedError
    
    @abstractmethod
    def validate_epsilon(self, epsilon: float) -> bool:
        """Validate epsilon parameter."""
        raise NotImplementedError


class ModelManager(Protocol):
    """Protocol for model resource management."""
    
    def __enter__(self) -> Tuple[Any, Any]:
        """Enter context and return loaded model and tokenizer."""
        raise NotImplementedError
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and cleanup resources."""
        raise NotImplementedError


class ConfigurableComponent(Protocol):
    """Protocol for configurable components."""
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure component with given parameters."""
        raise NotImplementedError
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        raise NotImplementedError


class DPFactory(Protocol):
    """Protocol for differential privacy mechanism factory."""
    
    def create_mechanism(self, mechanism_type: str, config: Dict[str, Any]) -> DPMechanism:
        """Create a differential privacy mechanism."""
        raise NotImplementedError
    
    def list_available_mechanisms(self) -> List[str]:
        """List available mechanism types."""
        raise NotImplementedError
