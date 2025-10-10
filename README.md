# DP-MLM: Differentially Private Text Rewriting

This is the fork of the original code repository for the ACL Findings paper: *DP-MLM: Differentially Private Text Rewriting Using Masked Language Models*

## Features

### 🏗️ **High-Level Architecture**
- **Modular Design**: Clean separation of concerns with dedicated modules
- **Type Safety**: Comprehensive type hints and Protocol-based interfaces
- **Plug-and-Play**: Factory pattern with preset configurations for easy usage
- **Resource Management**: Context managers for automatic cleanup
- **Multiple Mechanisms**: DPMLM, DPPrompt, DPParaphrase, and DPBart implementations

### 🚀 **Easy-to-Use Interface**
```python
import dpmlm

# Simple usage
mechanism = dpmlm.create_mechanism("dpmlm", epsilon=1.0)
result = mechanism.privatize("Your text here", epsilon=1.0)

# Using presets
mechanism = dpmlm.create_from_preset("dpmlm_high_privacy")
result = mechanism.privatize("Sensitive text")

# With PII annotation
mechanism = dpmlm.create_mechanism("dpmlm", annotator=pii_model)
result = mechanism.privatize("Text with names", epsilon=1.0)
```

### 📁 **Package Structure**
```
dpmlm/
├── __init__.py          # Package interface and exports
├── interfaces.py        # Abstract base classes and protocols
├── config.py           # Configuration classes with validation
├── resources.py        # Context managers for resource management
├── core.py             # DPMLM implementation
├── llmdp.py            # DPPrompt, DPParaphrase, DPBart implementations
└── factory.py          # Factory pattern for mechanism creation
```

## Setup
In this repository, you will find a `requirements.txt` file, which contains all necessary Python dependencies.

Firstly, clone the repository and install the required packages:

```bash
git clone https://github.com/yamaceay/dpmlm.git
cd dpmlm
pip install -r requirements.txt
```

Initialize the PII detection submodule:

```bash
make submodules
```

## Usage

### 🎯 **High-Level Interface (Recommended)**

The package provides a simple, type-safe interface for all differential privacy mechanisms:

```python
import dpmlm

# List available mechanisms and presets
print("Available mechanisms:", dpmlm.list_mechanisms())
print("Available presets:", dpmlm.list_presets())

# Create a mechanism
mechanism = dpmlm.create_mechanism(
    "dpmlm",
    config={
        "generic": {"device": "auto"},
        "model": {
            "model": {"model_name": "roberta-base"},
        },
    },
)

# Apply privacy
result = mechanism.privatize("Hello John Doe", epsilon=1.0)
print(f"Original: {result.original_text}")
print(f"Private: {result.private_text}")
print(f"Perturbed: {result.perturbed_tokens}/{result.total_tokens}")
```

### 📋 **Using Preset Configurations**

Choose from pre-configured settings optimized for different privacy needs:

```python
# High privacy for sensitive documents
mechanism = dpmlm.create_from_preset("dpmlm_high_privacy")

# PII-focused processing
mechanism = dpmlm.create_from_preset(
    "dpmlm_pii_focused",
    override_config={
        "model": {"annotator": pii_model},
    },
)

# Basic configuration
mechanism = dpmlm.create_from_preset("dpmlm_basic")

# Available presets:
# - dpmlm_basic, dpmlm_pii_focused, dpmlm_high_privacy
# - dpprompt_default, dpparaphrase_default  
# - dpbart_gaussian, dpbart_analytic
```

### 🧬 **Multiple DP Mechanisms**

The package supports various differential privacy mechanisms:

```python
# DP-MLM (masked language model approach)
dpmlm_mechanism = dpmlm.create_mechanism("dpmlm")

# DP-Prompt (prompt-based paraphrasing)
dpprompt_mechanism = dpmlm.create_mechanism("dpprompt")

# DP-Paraphrase (fine-tuned paraphrasing)
dpparaphrase_mechanism = dpmlm.create_mechanism("dpparaphrase")

# DP-Bart (BART-based with noise injection)
dpbart_mechanism = dpmlm.create_mechanism("dpbart")
```

### 🛡️ **PII-Aware Processing**

Integrate with PII detection models for selective privatization:

```python
# Load PII annotator
from pii import DataLabels, TorchTokenClassifier, PIIDeidentifier

labels = ['O'] + [f'B-{l}' for l in unique_labels] + [f'I-{l}' for l in unique_labels]
with TorchTokenClassifier('path/to/pii/model', DataLabels(labels)) as (model, tokenizer):
    annotator = PIIDeidentifier('outputs', model, tokenizer, DataLabels(labels))

# Create DPMLM with PII awareness
config = {
    "model": {
        "annotator": annotator,
    }
}
mechanism = dpmlm.create_mechanism("dpmlm", config=config)

result = mechanism.privatize("Mr. John Smith lives in New York", epsilon=1.0)
# Only names and locations will be privatized
```

### ⚙️ **Advanced Configuration**

Fine-tune mechanisms with type-safe configuration:

```python
# Build custom configuration
config = dpmlm.build_dpmlm_config(
    alpha=0.003,
    use_temperature=True,
    model={
        "model_name": "roberta-base",
        "max_sequence_length": 512,
    },
    add_probability=0.15,  # For "plus" method
    delete_probability=0.05
)

mechanism = dpmlm.create_mechanism("dpmlm", config=config)
result = mechanism.privatize(sample_text, epsilon=1.0)
```

## Requirements & Downloads

### Additional Dependencies
For `DPParaphrase` model, download the fine-tuned model directory from:
[Model Download Link](https://drive.google.com/drive/folders/1w_6MHQEw9LGkOHx_K1tc6t9djzrprITp?usp=sharing)

Download the WordNet 2022 corpus:
```bash
python -m wn download oewn:2022
```

### Important Notes
- Each implementation uses specific clipping bounds optimized for the paper's evaluation
- These bounds can be customized through configuration parameters for potentially better performance
- The high-level interface automatically handles optimal parameter selection
- PII detection models can be trained using the included `pii/` module

## Advanced Features

### 🔧 **Configuration Management**
The package provides type-safe configuration with automatic validation:

```python
# Validation catches common errors
config = dpmlm.DPMLMConfig(
    epsilon=-1.0  # Will raise ValueError: epsilon must be positive
)

# Get configuration suggestions
config = dpmlm.build_dpmlm_config(
    use_case="high_privacy",  # Automatically sets appropriate parameters
    model_name="roberta-base"
)
```

### 🚀 **Performance Optimization**
- Automatic device selection (MPS, CUDA, CPU)
- Context managers for efficient resource management
- Optimized token chunking for long documents
- Memory-efficient model loading and cleanup

### 📊 **Comprehensive Result Analysis**
```python
result = mechanism.privatize("Text with names", epsilon=1.0)

# Detailed statistics
print(f"Perturbation rate: {result.perturbation_rate:.2%}")
print(f"Privacy budget used: {result.epsilon}")
print(f"Added tokens: {result.added_tokens}")
print(f"Deleted tokens: {result.deleted_tokens}")
print(f"Processing time: {result.processing_time:.2f}s")
```

## Summary

This repository provides a comprehensive, production-ready implementation of differential privacy mechanisms for text processing. Key highlights:

- **🎯 Simple Interface**: One-line usage for common scenarios
- **🔧 Highly Configurable**: Type-safe configuration with validation  
- **🛡️ PII-Aware**: Intelligent selective privatization of sensitive information
- **⚡ Optimized**: Automatic resource management and device selection
- **📐 Multiple Mechanisms**: DPMLM, DPPrompt, DPParaphrase, and DPBart
- **🔌 Plug-and-Play**: Factory pattern with preset configurations
- **💻 CLI Ready**: Command-line interface for quick processing
- **📊 Detailed Analytics**: Comprehensive result reporting and statistics

Whether you're a researcher experimenting with differential privacy or a developer integrating privacy-preserving text processing into production systems, this package provides the tools you need with a clean, intuitive interface.

## Citation
Please consider citing the original work that introduced `DP-MLM`. Thank you!

```
@inproceedings{meisenbacher-etal-2024-dp,
    title = "{DP}-{MLM}: Differentially Private Text Rewriting Using Masked Language Models",
    author = "Meisenbacher, Stephen  and
      Chevli, Maulik  and
      Vladika, Juraj  and
      Matthes, Florian",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.554/",
    doi = "10.18653/v1/2024.findings-acl.554",
    pages = "9314--9328"
}
```
