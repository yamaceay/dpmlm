"""
Factory pattern for creating differential privacy mechanisms.

This module provides a unified interface for creating different types
of DP mechanisms with proper configuration management.
"""

import logging
from typing import Dict, Any, List, Optional
from dpmlm.interfaces import DPMechanism
from dpmlm.config import (
    DPMLMConfig, DPPromptConfig, DPParaphraseConfig, DPBartConfig,
    create_dpmlm_config, create_dpprompt_config, 
    create_dpparaphrase_config, create_dpbart_config
)

logger = logging.getLogger(__name__)


class DPMechanismFactory:
    """Factory for creating differential privacy mechanisms."""
    
    def __init__(self):
        self._creators = {
            "dpmlm": self._create_dpmlm,
            "dpprompt": self._create_dpprompt,
            "dpparaphrase": self._create_dpparaphrase,
            "dpbart": self._create_dpbart
        }
        
    def create_mechanism(self, mechanism_type: str, config: Optional[Dict[str, Any]] = None, 
                        annotator: Any = None) -> DPMechanism:
        """Create a differential privacy mechanism.
        
        Args:
            mechanism_type: Type of mechanism ("dpmlm", "dpprompt", "dpparaphrase", "dpbart")
            config: Configuration dictionary or None for defaults
            annotator: Optional annotator for PII detection (DPMLM only)
            
        Returns:
            Configured DPMechanism instance
            
        Raises:
            ValueError: If mechanism_type is not supported
        """
        if mechanism_type not in self._creators:
            raise ValueError(f"Unknown mechanism type: {mechanism_type}. "
                           f"Available types: {list(self._creators.keys())}")
        
        logger.info("Creating %s mechanism", mechanism_type)
        return self._creators[mechanism_type](config, annotator)
    
    def list_available_mechanisms(self) -> List[str]:
        """List available mechanism types."""
        return list(self._creators.keys())
    
    def _create_dpmlm(self, config: Optional[Dict[str, Any]], annotator: Any) -> DPMechanism:
        """Create DPMLM mechanism."""
        # Import locally to avoid circular imports and dependency issues
        try:
            from dpmlm.core import DPMLMPrivatizer
        except ImportError as e:
            logger.error("Failed to import DPMLMPrivatizer: %s", e)
            raise ImportError("DPMLM dependencies not available") from e
        
        if config is None:
            dp_config = DPMLMConfig()
        else:
            dp_config = create_dpmlm_config(**config)
        
        return DPMLMPrivatizer(dp_config, annotator)
    
    def _create_dpprompt(self, config: Optional[Dict[str, Any]], annotator: Any) -> DPMechanism:  # pylint: disable=unused-argument
        """Create DPPrompt mechanism."""
        try:
            from dpmlm.llmdp import DPPromptPrivatizer
        except ImportError as e:
            logger.error("Failed to import DPPromptPrivatizer: %s", e)
            raise ImportError("DPPrompt dependencies not available") from e
        
        if config is None:
            dp_config = DPPromptConfig()
        else:
            dp_config = create_dpprompt_config(**config)
        
        return DPPromptPrivatizer(dp_config)
    
    def _create_dpparaphrase(self, config: Optional[Dict[str, Any]], annotator: Any) -> DPMechanism:  # pylint: disable=unused-argument
        """Create DPParaphrase mechanism."""
        try:
            from dpmlm.llmdp import DPParaphrasePrivatizer
        except ImportError as e:
            logger.error("Failed to import DPParaphrasePrivatizer: %s", e)
            raise ImportError("DPParaphrase dependencies not available") from e
        
        if config is None:
            dp_config = DPParaphraseConfig()
        else:
            dp_config = create_dpparaphrase_config(**config)
        
        return DPParaphrasePrivatizer(dp_config)
    
    def _create_dpbart(self, config: Optional[Dict[str, Any]], annotator: Any) -> DPMechanism:  # pylint: disable=unused-argument
        """Create DPBart mechanism."""
        try:
            from dpmlm.llmdp import DPBartPrivatizer
        except ImportError as e:
            logger.error("Failed to import DPBartPrivatizer: %s", e)
            raise ImportError("DPBart dependencies not available") from e
        
        if config is None:
            dp_config = DPBartConfig()
        else:
            dp_config = create_dpbart_config(**config)
        
        return DPBartPrivatizer(dp_config)


class ConfigurableDPFactory:
    """Factory with preset configurations for common use cases."""
    
    def __init__(self):
        self.base_factory = DPMechanismFactory()
        self._presets = {
            "dpmlm_basic": {
                "type": "dpmlm",
                "config": {
                    "epsilon": 1.0,
                    "use_temperature": True,
                    "process_pii_only": False
                }
            },
            "dpmlm_pii_focused": {
                "type": "dpmlm", 
                "config": {
                    "epsilon": 1.0,
                    "use_temperature": True,
                    "process_pii_only": True
                }
            },
            "dpmlm_high_privacy": {
                "type": "dpmlm",
                "config": {
                    "epsilon": 0.1,
                    "use_temperature": True,
                    "process_pii_only": True
                }
            },
            "dpprompt_default": {
                "type": "dpprompt",
                "config": {
                    "epsilon": 1.0
                }
            },
            "dpparaphrase_default": {
                "type": "dpparaphrase", 
                "config": {
                    "epsilon": 1.0
                }
            },
            "dpbart_gaussian": {
                "type": "dpbart",
                "config": {
                    "epsilon": 1.0,
                    "noise_method": "gaussian"
                }
            },
            "dpbart_analytic": {
                "type": "dpbart",
                "config": {
                    "epsilon": 1.0,
                    "noise_method": "analytic_gaussian"
                }
            }
        }
    
    def create_from_preset(self, preset_name: str, annotator: Any = None, 
                          override_config: Optional[Dict[str, Any]] = None) -> DPMechanism:
        """Create mechanism from preset configuration.
        
        Args:
            preset_name: Name of preset configuration
            annotator: Optional annotator for PII detection
            override_config: Optional config overrides
            
        Returns:
            Configured DPMechanism instance
        """
        if preset_name not in self._presets:
            raise ValueError(f"Unknown preset: {preset_name}. "
                           f"Available presets: {list(self._presets.keys())}")
        
        preset = self._presets[preset_name].copy()
        mechanism_type = preset["type"]
        config = preset["config"].copy()
        
        # Apply overrides
        if override_config:
            config.update(override_config)
        
        logger.info("Creating mechanism from preset: %s", preset_name)
        return self.base_factory.create_mechanism(mechanism_type, config, annotator)
    
    def create_mechanism(self, mechanism_type: str, config: Optional[Dict[str, Any]] = None,
                        annotator: Any = None) -> DPMechanism:
        """Delegate to base factory."""
        return self.base_factory.create_mechanism(mechanism_type, config, annotator)
    
    def list_available_mechanisms(self) -> List[str]:
        """List available mechanism types."""
        return self.base_factory.list_available_mechanisms()
    
    def list_presets(self) -> List[str]:
        """List available preset configurations."""
        return list(self._presets.keys())


# Global factory instance
_factory = ConfigurableDPFactory()


def create_mechanism(mechanism_type: str, config: Optional[Dict[str, Any]] = None,
                    annotator: Any = None) -> DPMechanism:
    """Create a differential privacy mechanism using the global factory.
    
    Args:
        mechanism_type: Type of mechanism ("dpmlm", "dpprompt", "dpparaphrase", "dpbart")
        config: Configuration dictionary or None for defaults
        annotator: Optional annotator for PII detection (DPMLM only)
        
    Returns:
        Configured DPMechanism instance
    """
    return _factory.create_mechanism(mechanism_type, config, annotator)


def create_from_preset(preset_name: str, annotator: Any = None,
                      override_config: Optional[Dict[str, Any]] = None) -> DPMechanism:
    """Create mechanism from preset configuration using the global factory.
    
    Args:
        preset_name: Name of preset configuration
        annotator: Optional annotator for PII detection
        override_config: Optional config overrides
        
    Returns:
        Configured DPMechanism instance
    """
    return _factory.create_from_preset(preset_name, annotator, override_config)


def list_mechanisms() -> List[str]:
    """List available mechanism types."""
    return _factory.list_available_mechanisms()


def list_presets() -> List[str]:
    """List available preset configurations."""
    return _factory.list_presets()


# Configuration builders for common scenarios
def build_dpmlm_config(epsilon: float = 1.0, process_pii_only: bool = True,
                      model_name: str = "roberta-base", **kwargs) -> Dict[str, Any]:
    """Build DPMLM configuration with common parameters."""
    config = {
        "epsilon": epsilon,
        "process_pii_only": process_pii_only,
        "model_config": {
            "model_name": model_name
        }
    }
    config.update(kwargs)
    return config


def build_dpprompt_config(epsilon: float = 1.0, model_name: str = "google/flan-t5-base",
                         prompt_template: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Build DPPrompt configuration with common parameters."""
    config = {
        "epsilon": epsilon,
        "model_config": {
            "model_name": model_name
        }
    }
    if prompt_template:
        config["prompt_template"] = prompt_template
    config.update(kwargs)
    return config


def build_dpparaphrase_config(epsilon: float = 1.0, 
                             model_name: str = "models/gpt2-paraphraser",
                             **kwargs) -> Dict[str, Any]:
    """Build DPParaphrase configuration with common parameters."""
    config = {
        "epsilon": epsilon,
        "model_config": {
            "model_name": model_name
        }
    }
    config.update(kwargs)
    return config


def build_dpbart_config(epsilon: float = 1.0, model_name: str = "facebook/bart-base",
                       noise_method: str = "gaussian", **kwargs) -> Dict[str, Any]:
    """Build DPBart configuration with common parameters."""
    config = {
        "epsilon": epsilon,
        "noise_method": noise_method,
        "model_config": {
            "model_name": model_name
        }
    }
    config.update(kwargs)
    return config
