"""
Refactored LLMDP components with high-level, type-safe interfaces.

This module provides differential privacy mechanisms for language models
following the same design patterns as the DPMLM module.
"""

import logging
from typing import Optional, List, Dict, Any
import numpy as np
import torch
from pathlib import Path
from transformers import (
    LogitsProcessor, LogitsProcessorList, pipeline,
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    BartTokenizer, BartModel, BartForConditionalGeneration
)

from dpmlm.interfaces import DPMechanism, PrivacyResult
from dpmlm.config import DPPromptConfig, DPParaphraseConfig, DPBartConfig
from dpmlm.resources import (
    DPPromptModelManager, DPParaphraseModelManager, DPBartModelManager
)

logger = logging.getLogger(__name__)


class ClipLogitsProcessor(LogitsProcessor):
    """Logits processor for clipping values to specified range."""
    
    def __init__(self, min_value: float = -100, max_value: float = 100):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply clipping to logits."""
        return torch.clamp(scores, min=self.min_value, max=self.max_value)


class DPPromptPrivatizer(DPMechanism):
    """High-level DPPrompt privatizer for T5-based models."""
    
    def __init__(self, config: Optional[DPPromptConfig] = None):
        self.config = config or DPPromptConfig()
        self._setup_logits_processor()
        
    def _setup_logits_processor(self) -> None:
        """Setup logits processor for clipping."""
        self.logits_processor = LogitsProcessorList([
            ClipLogitsProcessor(self.config.min_logit, self.config.max_logit)
        ])
        
    def validate_epsilon(self, epsilon: float) -> bool:
        """Validate epsilon parameter."""
        return epsilon > 0
    
    def _calculate_temperature(self, epsilon: float) -> float:
        """Calculate temperature for differential privacy."""
        return 2 * self.config.sensitivity / epsilon
    
    def _create_prompt(self, text: str) -> str:
        """Create prompt from template."""
        return self.config.prompt_template.format(text)
    
    def privatize(self, text: str, epsilon: float, **kwargs) -> PrivacyResult:
        """Apply differential privacy using prompt-based approach."""
        if not self.validate_epsilon(epsilon):
            raise ValueError(f"Invalid epsilon: {epsilon}")
        
        temperature = self._calculate_temperature(epsilon)
        prompt = self._create_prompt(text)
        
        with DPPromptModelManager(self.config.model_config, self.config.device) as (model, tokenizer):
            
            model_inputs = tokenizer(
                prompt, 
                max_length=self.config.model_config.max_sequence_length,
                truncation=self.config.model_config.truncation,
                return_tensors="pt"
            ).to(model.device)
            
            
            max_new_tokens = (
                self.config.max_new_tokens or 
                len(model_inputs["input_ids"][0])
            )
            
            
            output = model.generate(
                **model_inputs,
                do_sample=self.config.do_sample,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                logits_processor=self.logits_processor
            )
            
            private_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            
            if private_text.startswith(prompt):
                private_text = private_text[len(prompt):].strip()
        
        
        original_tokens = len(text.split())
        private_tokens = len(private_text.split())
        
        return PrivacyResult(
            original_text=text,
            private_text=private_text,
            perturbed_tokens=abs(private_tokens - original_tokens),
            total_tokens=original_tokens,
            metadata={
                "method": "prompt",
                "temperature": temperature,
                "epsilon": epsilon
            }
        )


class DPParaphrasePrivatizer(DPMechanism):
    """High-level DPParaphrase privatizer for GPT2-based models."""
    
    def __init__(self, config: Optional[DPParaphraseConfig] = None):
        self.config = config or DPParaphraseConfig()
        self._setup_logits_processor()
        
    def _setup_logits_processor(self) -> None:
        """Setup logits processor for clipping."""
        self.logits_processor = LogitsProcessorList([
            ClipLogitsProcessor(self.config.min_logit, self.config.max_logit)
        ])
        
    def validate_epsilon(self, epsilon: float) -> bool:
        """Validate epsilon parameter."""
        return epsilon > 0
    
    def _calculate_temperature(self, epsilon: float) -> float:
        """Calculate temperature for differential privacy."""
        return 2 * self.config.sensitivity / epsilon
    
    def _create_prompt(self, text: str) -> str:
        """Create prompt with suffix."""
        return text + self.config.prompt_suffix
    
    def _cleanup_output(self, text: str, prompt: str) -> str:
        """Clean up generated output."""
        
        result = text.replace(prompt, "").replace(prompt.strip(), "")
        
        
        for char in self.config.output_cleanup_chars:
            result = result.replace(char, " " if char == "\xa0" else "")
        
        return result.strip()
    
    def privatize(self, text: str, epsilon: float, **kwargs) -> PrivacyResult:
        """Apply differential privacy using paraphrase-based approach."""
        if not self.validate_epsilon(epsilon):
            raise ValueError(f"Invalid epsilon: {epsilon}")
        
        temperature = self._calculate_temperature(epsilon)
        prompt = self._create_prompt(text)
        
        with DPParaphraseModelManager(self.config.model_config, self.config.device) as (model, tokenizer):
            
            pipeline_device = str(model.device) if hasattr(model, "device") else self.config.device

            pipe = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                logits_processor=self.logits_processor,
                device=pipeline_device,
                pad_token_id=tokenizer.eos_token_id
            )
            
            
            length = len(tokenizer(prompt)["input_ids"])
            max_new_tokens = self.config.max_new_tokens or length
            
            
            generated = pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )[0]["generated_text"]
            
            
            private_text = self._cleanup_output(generated, prompt)
        
        
        original_tokens = len(text.split())
        private_tokens = len(private_text.split())
        
        return PrivacyResult(
            original_text=text,
            private_text=private_text,
            perturbed_tokens=abs(private_tokens - original_tokens),
            total_tokens=original_tokens,
            metadata={
                "method": "paraphrase",
                "temperature": temperature,
                "epsilon": epsilon
            }
        )


class DPBartPrivatizer(DPMechanism):
    """High-level DPBart privatizer for BART-based models."""
    
    def __init__(self, config: Optional[DPBartConfig] = None):
        self.config = config or DPBartConfig()
        
    def validate_epsilon(self, epsilon: float) -> bool:
        """Validate epsilon parameter."""
        return epsilon > 0
    
    def _clip_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """Clip vector to specified range."""
        return torch.clip(vector, self.config.c_min, self.config.c_max)
    
    def _calibrate_analytic_gaussian(self, epsilon: float, delta: float, sensitivity: float) -> float:
        """Calibrate analytic Gaussian mechanism with high precision."""
        
        try:
            import mpmath
            from mpmath import mp
        except ImportError:
            logger.warning("mpmath not available, using standard Gaussian calibration")
            scale = np.sqrt((sensitivity**2 / epsilon**2) * 2 * np.log(1.25 / delta))
            return scale
        
        
        if epsilon <= 1000:
            mp.dps = self.config.precision_dps.get("1000", 500)
        elif epsilon <= 2500:
            mp.dps = self.config.precision_dps.get("2500", 1100)
        else:
            mp.dps = self.config.precision_dps.get("5000", 2200)
        
        def phi(t):
            return 0.5 * (1.0 + mpmath.erf(t / mpmath.sqrt(2.0)))
        
        def case_a(eps, s):
            return phi(mpmath.sqrt(eps * s)) - mpmath.exp(eps) * phi(-mpmath.sqrt(eps * (s + 2.0)))
        
        def case_b(eps, s):
            return phi(-mpmath.sqrt(eps * s)) - mpmath.exp(eps) * phi(-mpmath.sqrt(eps * (s + 2.0)))
        
        def doubling_trick(predicate_stop, s_inf, s_sup):
            while not predicate_stop(s_sup):
                s_inf = s_sup
                s_sup = 2.0 * s_inf
            return s_inf, s_sup
        
        def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
            s_mid = s_inf + (s_sup - s_inf) / 2.0
            while not predicate_stop(s_mid):
                if predicate_left(s_mid):
                    s_sup = s_mid
                else:
                    s_inf = s_mid
                s_mid = s_inf + (s_sup - s_inf) / 2.0
            return s_mid
        
        delta_threshold = case_a(epsilon, 0.0)
        
        if delta == delta_threshold:
            alpha = 1.0
        else:
            if delta > delta_threshold:
                predicate_stop_dt = lambda s: case_a(epsilon, s) >= delta
                function_s_to_delta = lambda s: case_a(epsilon, s)
                predicate_left_bs = lambda s: function_s_to_delta(s) > delta
                function_s_to_alpha = lambda s: mpmath.sqrt(1.0 + s / 2.0) - mpmath.sqrt(s / 2.0)
            else:
                predicate_stop_dt = lambda s: case_b(epsilon, s) <= delta
                function_s_to_delta = lambda s: case_b(epsilon, s)
                predicate_left_bs = lambda s: function_s_to_delta(s) < delta
                function_s_to_alpha = lambda s: mpmath.sqrt(1.0 + s / 2.0) + mpmath.sqrt(s / 2.0)
            
            predicate_stop_bs = lambda s: abs(function_s_to_delta(s) - delta) <= self.config.analytic_tolerance
            
            s_inf, s_sup = doubling_trick(predicate_stop_dt, 0.0, 1.0)
            s_final = binary_search(predicate_stop_bs, predicate_left_bs, s_inf, s_sup)
            alpha = function_s_to_alpha(s_final)
        
        sigma = alpha * sensitivity / mpmath.sqrt(2.0 * epsilon)
        return float(sigma)
    
    def _add_noise(self, vector: torch.Tensor, epsilon: float, method: Optional[str] = None) -> torch.Tensor:
        """Add differential privacy noise to vector."""
        k = vector.shape[-1]
        noise_method = method or self.config.noise_method
        
        if noise_method == "laplace":
            sensitivity = 2 * self.config.sigma * self.config.num_sigmas * k
            noise = torch.from_numpy(np.random.laplace(0, sensitivity / epsilon, size=k))
        elif noise_method == "gaussian":
            sensitivity = 2 * self.config.sigma * self.config.num_sigmas * np.sqrt(k)
            scale = np.sqrt((sensitivity**2 / epsilon**2) * 2 * np.log(1.25 / self.config.delta))
            noise = torch.from_numpy(np.random.normal(0, scale, size=k))
        elif noise_method == "analytic_gaussian":
            sensitivity = 2 * self.config.sigma * self.config.num_sigmas * np.sqrt(k)
            analytic_scale = self._calibrate_analytic_gaussian(epsilon, self.config.delta, sensitivity)
            noise = torch.from_numpy(np.random.normal(0, analytic_scale, size=k))
        else:
            raise ValueError(f"Unknown noise method: {noise_method}")
        
        return vector + noise
    
    def privatize(self, text: str, epsilon: float, **kwargs) -> PrivacyResult:
        """Apply differential privacy using BART-based approach."""
        if not self.validate_epsilon(epsilon):
            raise ValueError(f"Invalid epsilon: {epsilon}")
        
        method = kwargs.get('noise_method', self.config.noise_method)
        
        with DPBartModelManager(self.config.model_config, self.config.device) as (models, tokenizer):
            encoder_model, decoder_model = models
            
            
            inputs = tokenizer(
                text,
                max_length=self.config.model_config.max_sequence_length,
                truncation=self.config.model_config.truncation,
                return_tensors="pt"
            ).to(encoder_model.device)
            
            num_tokens = len(inputs["input_ids"][0])
            
            
            enc_output = encoder_model.encoder(**inputs)
            
            
            clipped_hidden = self._clip_vector(enc_output["last_hidden_state"].cpu())
            noisy_hidden = self._add_noise(clipped_hidden, epsilon, method)
            enc_output["last_hidden_state"] = noisy_hidden.float().to(encoder_model.device)
            
            
            dec_out = decoder_model.generate(
                encoder_outputs=enc_output,
                max_new_tokens=num_tokens
            )
            
            private_text = tokenizer.decode(dec_out[0], skip_special_tokens=True).strip()
        
        
        original_tokens = len(text.split())
        private_tokens = len(private_text.split())
        
        return PrivacyResult(
            original_text=text,
            private_text=private_text,
            perturbed_tokens=abs(private_tokens - original_tokens),
            total_tokens=original_tokens,
            metadata={
                "method": "bart",
                "noise_method": method,
                "epsilon": epsilon,
                "sigma": self.config.sigma,
                "delta": self.config.delta
            }
        )



def create_dpprompt_privatizer(config: Optional[DPPromptConfig] = None) -> DPPromptPrivatizer:
    """Create a DPPrompt privatizer with optional configuration."""
    return DPPromptPrivatizer(config)

def create_dpparaphrase_privatizer(config: Optional[DPParaphraseConfig] = None) -> DPParaphrasePrivatizer:
    """Create a DPParaphrase privatizer with optional configuration."""
    return DPParaphrasePrivatizer(config)

def create_dpbart_privatizer(config: Optional[DPBartConfig] = None) -> DPBartPrivatizer:
    """Create a DPBart privatizer with optional configuration."""
    return DPBartPrivatizer(config)
