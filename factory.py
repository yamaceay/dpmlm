"""
Factory pattern for creating differential privacy mechanisms.

This module provides a unified interface for creating different types
of DP mechanisms with proper configuration management.
"""

import copy
import logging
from typing import Dict, Any, List, Optional
from dpmlm.interfaces import DPMechanism
from dpmlm.config import (
    DPMLMConfig,
    DPPromptConfig,
    DPParaphraseConfig,
    DPBartConfig,
)
from dpmlm.core import DPMLMMechanism

logger = logging.getLogger(__name__)


class DPMechanismFactory:
    """Factory for creating differential privacy mechanisms."""
    
    def __init__(self):
        self._creators = {
            "dpmlm": self._create_dpmlm,
            "dpprompt": self._create_dpprompt,
            "dpparaphrase": self._create_dpparaphrase,
            "dpbart": self._create_dpbart,
        }
        
    def create_mechanism(self, mechanism_type: str, config: Optional[Dict[str, Any]] = None) -> DPMechanism:
        """Create a differential privacy mechanism.
        
        Args:
            mechanism_type: Type of mechanism ("dpmlm", "dpmlm_riskaware",
                "dpprompt", "dpparaphrase", "dpbart")
            config: Configuration dictionary or None for defaults
            
        Returns:
            Configured DPMechanism instance
            
        Raises:
            ValueError: If mechanism_type is not supported
        """
        if mechanism_type not in self._creators:
            raise ValueError(f"Unknown mechanism type: {mechanism_type}. "
                           f"Available types: {list(self._creators.keys())}")
        
        logger.info("Creating %s mechanism", mechanism_type)
        return self._creators[mechanism_type](config)
    
    def list_available_mechanisms(self) -> List[str]:
        """List available mechanism types."""
        return list(self._creators.keys())

    @staticmethod
    def _normalize_llmdp_payload(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Flatten structured LLMDP configuration sections into a single mapping."""
        config_dict = dict(config or {})
        has_sections = any(
            isinstance(config_dict.get(section), dict)
            for section in ("generic", "model", "runtime")
        )
        if not has_sections:
            return config_dict

        flattened: Dict[str, Any] = {}
        for section in ("generic", "model", "runtime"):
            section_payload = config_dict.get(section)
            if isinstance(section_payload, dict):
                flattened.update(section_payload)

        for key, value in config_dict.items():
            if key not in {"generic", "model", "runtime"} and key not in flattened:
                flattened[key] = value

        return flattened

    def _create_dpmlm(self, config: Optional[Dict[str, Any]]) -> DPMechanism:
        """Create risk-aware DPMLM rewrite mechanism."""

        if isinstance(config, DPMLMConfig):
            dp_config = config
        else:
            config_dict = dict(config or {})

            generic_payload = config_dict.get("generic", {})
            runtime_payload = config_dict.get("runtime", {})

            if any(
                key in config_dict
                for key in ("dpmlm", "dpmlm_config", "risk", "risk_settings", "risk_config")
            ):
                model_payload = config_dict.get("dpmlm") or config_dict.get("dpmlm_config") or {}
                risk_payload = (
                    config_dict.get("risk")
                    or config_dict.get("risk_settings")
                    or config_dict.get("risk_config")
                    or {}
                )
                model_payload = {**model_payload, **risk_payload}
            else:
                model_payload = config_dict.get("model") or {
                    key: value
                    for key, value in config_dict.items()
                    if key not in {"generic", "runtime"}
                }

            dp_config = DPMLMConfig(
                generic=generic_payload,
                model=model_payload,
                runtime=runtime_payload,
            )

        return DPMLMMechanism(config=dp_config)

    def _create_dpprompt(self, config: Optional[Dict[str, Any]]) -> DPMechanism:  
        """Create DPPrompt mechanism."""
        try:
            from dpmlm.llmdp import DPPromptPrivatizer
        except ImportError as e:
            logger.error("Failed to import DPPromptPrivatizer: %s", e)
            raise ImportError("DPPrompt dependencies not available") from e

        if isinstance(config, DPPromptConfig):
            dp_config = config
        else:
            payload = self._normalize_llmdp_payload(config)
            dp_config = DPPromptConfig(**payload)

        return DPPromptPrivatizer(dp_config)
    
    def _create_dpparaphrase(self, config: Optional[Dict[str, Any]]) -> DPMechanism:  
        """Create DPParaphrase mechanism."""
        try:
            from dpmlm.llmdp import DPParaphrasePrivatizer
        except ImportError as e:
            logger.error("Failed to import DPParaphrasePrivatizer: %s", e)
            raise ImportError("DPParaphrase dependencies not available") from e
        
        if isinstance(config, DPParaphraseConfig):
            dp_config = config
        else:
            payload = self._normalize_llmdp_payload(config)
            dp_config = DPParaphraseConfig(**payload)
        
        return DPParaphrasePrivatizer(dp_config)
    
    def _create_dpbart(self, config: Optional[Dict[str, Any]]) -> DPMechanism:  
        """Create DPBart mechanism."""
        try:
            from dpmlm.llmdp import DPBartPrivatizer
        except ImportError as e:
            logger.error("Failed to import DPBartPrivatizer: %s", e)
            raise ImportError("DPBart dependencies not available") from e
        
        if isinstance(config, DPBartConfig):
            dp_config = config
        else:
            payload = self._normalize_llmdp_payload(config)
            dp_config = DPBartConfig(**payload)
        
        return DPBartPrivatizer(dp_config)


class ConfigurableDPFactory:
    """Factory with preset configurations for common use cases."""
    
    def __init__(self):
        self.base_factory = DPMechanismFactory()
        self._presets = {
            "dpmlm_basic": {
                "type": "dpmlm",
                "config": {
                    "model": {
                        "use_temperature": True,
                        "process_pii_only": False,
                    }
                }
            },
            "dpmlm_pii_focused": {
                "type": "dpmlm", 
                "config": {
                    "model": {
                        "use_temperature": True,
                        "process_pii_only": True,
                    }
                }
            },
            "dpmlm_high_privacy": {
                "type": "dpmlm",
                "config": {
                    "model": {
                        "use_temperature": True,
                        "process_pii_only": True,
                        "add_probability": 0.0,
                        "delete_probability": 0.0,
                    }
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
    
    def create_from_preset(self, preset_name: str,
                          override_config: Optional[Dict[str, Any]] = None) -> DPMechanism:
        """Create mechanism from preset configuration.
        
        Args:
            preset_name: Name of preset configuration
            override_config: Optional config overrides
            
        Returns:
            Configured DPMechanism instance
        """
        if preset_name not in self._presets:
            raise ValueError(f"Unknown preset: {preset_name}. "
                           f"Available presets: {list(self._presets.keys())}")
        
        preset = self._presets[preset_name].copy()
        mechanism_type = preset["type"]
        config = copy.deepcopy(preset["config"])

        if override_config:
            for key, value in override_config.items():
                if key in {"model", "generic", "runtime"} and isinstance(value, dict):
                    section = config.setdefault(key, {})
                    section.update(value)
                else:
                    config[key] = value
        
        logger.info("Creating mechanism from preset: %s", preset_name)
        return self.base_factory.create_mechanism(mechanism_type, config)

    def create_mechanism(self, mechanism_type: str, config: Optional[Dict[str, Any]] = None) -> DPMechanism:
        """Delegate to base factory."""
        return self.base_factory.create_mechanism(mechanism_type, config)
    
    def list_available_mechanisms(self) -> List[str]:
        """List available mechanism types."""
        return self.base_factory.list_available_mechanisms()
    
    def list_presets(self) -> List[str]:
        """List available preset configurations."""
        return list(self._presets.keys())



_factory = ConfigurableDPFactory()


def create_mechanism(mechanism_type: str, config: Optional[Dict[str, Any]] = None) -> DPMechanism:
    """Create a differential privacy mechanism using the global factory.
    
    Args:
        mechanism_type: Type of mechanism ("dpmlm", "dpprompt", "dpparaphrase", "dpbart")
        config: Configuration dictionary or None for defaults
        
    Returns:
        Configured DPMechanism instance
    """
    return _factory.create_mechanism(mechanism_type, config)


def create_from_preset(preset_name: str,
                      override_config: Optional[Dict[str, Any]] = None) -> DPMechanism:
    """Create mechanism from preset configuration using the global factory.
    
    Args:
        preset_name: Name of preset configuration
        override_config: Optional config overrides
        
    Returns:
        Configured DPMechanism instance
    """
    return _factory.create_from_preset(preset_name, override_config)


def list_mechanisms() -> List[str]:
    """List available mechanism types."""
    return _factory.list_available_mechanisms()


def list_presets() -> List[str]:
    """List available preset configurations."""
    return _factory.list_presets()



def build_dpmlm_config(
    process_pii_only: bool = True,
    model_name: str = "roberta-base",
    **model_overrides: Any,
) -> Dict[str, Any]:
    """Build a DPMLM configuration payload using the new categorical layout."""

    model_section: Dict[str, Any] = {
        "process_pii_only": process_pii_only,
        "model": {
            "model_name": model_name,
        },
    }

    transformer_overrides = model_overrides.pop("model", None) or model_overrides.pop(
        "model_config", None
    )
    if transformer_overrides:
        if not isinstance(transformer_overrides, dict):
            raise TypeError("model overrides must be provided as a dictionary")
        model_section["model"].update(transformer_overrides)

    model_section.update(model_overrides)
    return {"model": model_section}


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
