"""
DPMLM package - High-level differential privacy for text processing.

This package provides type-safe, plug-and-play components for differential
privacy text processing with clean interfaces and resource management.
"""

from dpmlm.interfaces import (
    PrivacyResult, 
    TokenSpan, 
    DPMechanism,
    DPTokenizer,
    PIIAnnotator,
    DPModel
)

from dpmlm.config import (
    DPMLMConfig,
    DPPromptConfig, 
    DPParaphraseConfig,
    DPBartConfig,
    AnnotatorConfig,
    create_dpmlm_config,
    create_dpprompt_config,
    create_dpparaphrase_config,
    create_dpbart_config,
    create_annotator_config
)

from dpmlm.factory import (
    create_mechanism,
    create_from_preset,
    list_mechanisms,
    list_presets,
    build_dpmlm_config,
    build_dpprompt_config,
    build_dpparaphrase_config,
    build_dpbart_config
)

__version__ = "0.1.0"
__all__ = [
    # Core interfaces
    "PrivacyResult",
    "TokenSpan", 
    "DPMechanism",
    "DPTokenizer",
    "PIIAnnotator", 
    "DPModel",
    
    # Configuration
    "DPMLMConfig",
    "DPPromptConfig",
    "DPParaphraseConfig", 
    "DPBartConfig",
    "AnnotatorConfig",
    "create_dpmlm_config",
    "create_dpprompt_config",
    "create_dpparaphrase_config",
    "create_dpbart_config",
    "create_annotator_config",
    
    # Factory functions
    "create_mechanism",
    "create_from_preset", 
    "list_mechanisms",
    "list_presets",
    "build_dpmlm_config",
    "build_dpprompt_config",
    "build_dpparaphrase_config", 
    "build_dpbart_config"
]
