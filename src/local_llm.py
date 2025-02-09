from typing import Any, List, Optional, ClassVar
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import json
import re
import os

# Set PyTorch memory allocation to use expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Initialize model and tokenizer at module level
_model = None
_tokenizer = None

def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        _model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
            max_memory={0: "20GiB", "cpu": "32GiB"}
        )
    return _model, _tokenizer

class LocalDeepSeekLLM(LLM):
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    temperature: float = 0.1  # Lower temperature for more deterministic outputs
    model: ClassVar = None
    tokenizer: ClassVar = None
    
    def __init__(self):
        super().__init__()
        # Get existing model and tokenizer instances
        self.__class__.model, self.__class__.tokenizer = get_model_and_tokenizer()
    
    @property
    def _llm_type(self) -> str:
        return "local_deepseek"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                temperature=self.temperature,
                do_sample=True,
                max_new_tokens=2048,  # Ensure we get complete responses
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 1) Decode the raw response
        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 2) Remove the prompt from the response
        raw_response = raw_response[len(prompt):].strip()
        
        # 3) Remove everything from the start until the first "</think>" (including "</think>")
        cleaned_after_think = re.sub(r'^.*?</think>', '', raw_response, count=1, flags=re.DOTALL).strip()
        
        # 4) Try to find and parse JSON in the response
        try:
            # Look for JSON-like structure
            json_match = re.search(r'\{.*\}', cleaned_after_think, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                # Handle different JSON structures we expect
                if "queries" in parsed:
                    return json.dumps(parsed)  # Return full queries object
                elif "sections" in parsed:
                    return json.dumps(parsed)  # Return full sections object
                elif "final_answer" in parsed:
                    return parsed["final_answer"].strip()
                else:
                    return json.dumps(parsed)  # Return whatever JSON we found
            else:
                return cleaned_after_think
        except (json.JSONDecodeError, TypeError, AttributeError):
            # If JSON parsing fails, return the cleaned text as-is
            return cleaned_after_think