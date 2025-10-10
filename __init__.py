"""
DPMLM package - High-level differential privacy for text processing.

This package provides type-safe, plug-and-play components for differential
privacy text processing with clean interfaces and resource management.
Includes explainability-aware DP mechanisms for adaptive privacy protection.
"""

from dpmlm.factory import (
    list_mechanisms, list_presets,
    create_from_preset, create_mechanism
)