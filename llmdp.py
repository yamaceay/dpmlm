# class code for LLM-based Differential Privacy mechanisms

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList, pipeline
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
import torch
import numpy as np
from pathlib import Path

import mpmath
from mpmath import mp

from dataclasses import dataclass

import nltk
nltk.download('punkt_tab', quiet=True)

from dpmlm.interfaces import PrivacyResult

class ClipLogitsProcessor(LogitsProcessor):
  def __init__(self, min=-100, max=100):
    self.min = min
    self.max = max

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    scores = torch.clamp(scores, min=self.min, max=self.max)
    return scores
  
@dataclass
class DPPromptPrivatizer:
    model_checkpoint="google/flan-t5-base"
    min_logit=-19.22705113016047
    max_logit=7.48324937989716
    max_length=512
    device=None

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint).to(self.device)

        self.sensitivity = abs(self.max_logit - self.min_logit)
        self.logits_processor = LogitsProcessorList([ClipLogitsProcessor(self.min_logit, self.max_logit)])


    def prompt_template_fn(self, doc):
        prompt = "Document : {}\nParaphrase of the document :".format(doc)
        return prompt
    
    def privatize(self, text, epsilon=100):
        temperature = 2 * self.sensitivity / epsilon

        prompt = self.prompt_template_fn(text)

        model_inputs = self.tokenizer(prompt, max_length=self.max_length, truncation=True, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **model_inputs,
            do_sample = True,
            top_k=0,
            top_p=1.0,
            temperature=temperature,
            max_new_tokens=len(model_inputs["input_ids"][0]),
            logits_processor=self.logits_processor)

        private_text = self.tokenizer.decode(output[0], skip_special_tokens = True )
        return PrivacyResult(
            original_text=prompt,
            private_text=private_text,
            perturbed_tokens=0,
            total_tokens=0,
            added_tokens=0,
            deleted_tokens=0,
            metadata={"epsilon": epsilon}
        )

@dataclass
class DPParaphrasePrivatizer():
    model_checkpoint = "./models/gpt2-paraphraser"
    min_logit=-96.85249956065758
    max_logit=-8.747697966442914
    device=None

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_checkpoint, pad_token_id=self.tokenizer.eos_token_id).to(self.device)

        self.sensitivity = abs(self.max_logit - self.min_logit)
        self.logits_processor = LogitsProcessorList([ClipLogitsProcessor(self.min_logit, self.max_logit)])

        self.pipe = pipeline(task="text-generation", model=self.model, 
                             tokenizer=self.tokenizer, 
                             logits_processor=self.logits_processor, 
                             device=self.device,
                             pad_token_id=self.tokenizer.eos_token_id)

    def privatize(self, text, epsilon=100):
        temperature = 2 * self.sensitivity / epsilon

        prompt = text+ " >>>>> "
        length = len(self.tokenizer(prompt)["input_ids"])
    
        private_text = self.pipe(prompt, max_new_tokens=length, temperature=temperature)[0]["generated_text"]
        private_text = private_text.replace(prompt, "").replace(prompt.strip(), "").replace(u'\xa0', u' ').replace(">", "").strip()

        return PrivacyResult(
            original_text=prompt,
            private_text=private_text,
            perturbed_tokens=0,
            total_tokens=0,
            added_tokens=0,
            deleted_tokens=0,
            metadata={"epsilon": epsilon}
        )

@dataclass
class DPBartPrivatizer():
    sigma = 0.2
    num_sigmas=1/2
    delta = 1e-5
    max_length=512
    device=None

    def __post_init__(self, model='facebook/bart-base', ):
        self.tokenizer = BartTokenizer.from_pretrained(model)
        self.model = BartModel.from_pretrained(model).to(self.device)
        self.decoder = BartForConditionalGeneration.from_pretrained(model).to(self.device)

        self.c_min = -self.sigma
        self.c_max = self.num_sigmas * self.sigma

    def clip(self, vector):
        return torch.clip(vector, self.c_min, self.c_max)

    # https://github.com/trusthlt/dp-bart-private-rewriting/
    def calibrateAnalyticGaussianMechanism_precision(self, epsilon, delta, GS, tol = 1.e-12):
        """
        High-precision version of `calibrateAnalyticGaussianMechanism`.
        Should not be used for epsilon values above 5000.
        """

        if epsilon <= 1000:
            mp.dps = 500  # works for epsilon = 1000
        elif epsilon <= 2500 and epsilon > 1000:
            mp.dps = 1100  # works for epsilon = 2500
        else:
            mp.dps = 2200  # works for epsilon = 5000

        def Phi(t):
            return 0.5*(1.0 + mpmath.erf(t/mpmath.sqrt(2.0)))

        def caseA(epsilon,s):
            return Phi(mpmath.sqrt(epsilon*s)) - mpmath.exp(epsilon)*Phi(-mpmath.sqrt(epsilon*(s+2.0)))

        def caseB(epsilon,s):
            return Phi(-mpmath.sqrt(epsilon*s)) - mpmath.exp(epsilon)*Phi(-mpmath.sqrt(epsilon*(s+2.0)))

        def doubling_trick(predicate_stop, s_inf, s_sup):
            while(not predicate_stop(s_sup)):
                s_inf = s_sup
                s_sup = 2.0*s_inf
            return s_inf, s_sup

        def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
            s_mid = s_inf + (s_sup-s_inf)/2.0
            while(not predicate_stop(s_mid)):
                if (predicate_left(s_mid)):
                    s_sup = s_mid
                else:
                    s_inf = s_mid
                s_mid = s_inf + (s_sup-s_inf)/2.0
            return s_mid

        delta_thr = caseA(epsilon, 0.0)

        if (delta == delta_thr):
            alpha = 1.0

        else:
            if (delta > delta_thr):
                predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
                function_s_to_delta = lambda s : caseA(epsilon, s)
                predicate_left_BS = lambda s : function_s_to_delta(s) > delta
                function_s_to_alpha = lambda s : mpmath.sqrt(1.0 + s/2.0) - mpmath.sqrt(s/2.0)

            else:
                predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
                function_s_to_delta = lambda s : caseB(epsilon, s)
                predicate_left_BS = lambda s : function_s_to_delta(s) < delta
                function_s_to_alpha = lambda s : mpmath.sqrt(1.0 + s/2.0) + mpmath.sqrt(s/2.0)

            predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

            s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
            s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
            alpha = function_s_to_alpha(s_final)

        sigma = alpha*GS/mpmath.sqrt(2.0*epsilon)

        return float(sigma)
    
    def noise(self, vector, epsilon, delta=1e-5, method="gaussian"):
        k = vector.shape[-1]
        if method == "laplace":
            sensitivity = 2 * self.sigma * self.num_sigmas * k
            Z = torch.from_numpy(np.random.laplace(0, sensitivity / epsilon, size=k))
        elif method == 'gaussian':
            sensitivity = 2 * self.sigma * self.num_sigmas * np.sqrt(k)
            scale = np.sqrt((sensitivity**2 / epsilon**2) * 2 * np.log(1.25 / self.delta))
            Z = torch.from_numpy(np.random.normal(0, scale, size=k))
        elif method == "analytic_gaussian":
            sensitivity = 2 * self.sigma * self.num_sigmas * np.sqrt(k)
            analytic_scale = self.calibrateAnalyticGaussianMechanism_precision(epsilon, self.delta, sensitivity)
            Z = torch.from_numpy(np.random.normal(0, analytic_scale, size=k))
        return vector + Z

    def privatize(self, text, epsilon=100, method="gaussian"):
        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors="pt").to(self.device)
        num_tokens = len(inputs["input_ids"][0])

        enc_output = self.model.encoder(**inputs)
        enc_output["last_hidden_state"] = self.noise(self.clip(enc_output["last_hidden_state"].cpu()), epsilon=epsilon, delta=self.delta, method=method).float().to(self.device)

        dec_out = self.decoder.generate(encoder_outputs=enc_output, max_new_tokens=num_tokens)
        private_text = self.tokenizer.decode(dec_out[0], skip_special_tokens=True)
        private_text = private_text.strip()

        return PrivacyResult(
            original_text=text,
            private_text=private_text,
            perturbed_tokens=0,
            total_tokens=0,
            added_tokens=0,
            deleted_tokens=0,
            metadata={"epsilon": epsilon}
        )