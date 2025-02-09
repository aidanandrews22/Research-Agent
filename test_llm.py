import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Set PyTorch memory allocation to use expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    device_map="auto",  # Changed from "cuda" to "auto" for better memory management
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # Updated from use_flash_attention_2
    torch_dtype=torch.float16,
    max_memory={0: "20GiB", "cpu": "32GiB"}  # Explicit memory management
)

# Test the model
def test_model():
    prompt = "What is machine learning?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with some reasonable parameters
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

if __name__ == "__main__":
    test_model()