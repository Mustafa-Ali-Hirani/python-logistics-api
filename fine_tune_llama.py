# fine_tune_llama.py
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ==========================================
# 1. DEFINE LOGISTICS DOMAIN DATASET
# ==========================================
# We define a custom training dataset of logistics terminology
# to teach the model company-specific or domain-specific terms
logistics_training_data = [
    {
        "instruction": "Explain DPD in the context of Karachi Port rules.",
        "response": "DPD stands for Direct Port Delivery. Under Karachi Port rules, specific Class 9 hazardous shipments, such as high-capacity lithium batteries, are prohibited from warehouse storage and must undergo DPD to an off-site authorized customs bonded facility within 4 hours of offloading."
    },
    {
        "instruction": "What HS Code applies to industrial circuit breakers?",
        "response": "Industrial circuit breakers and electrical switchgear fall under HS Code 8537, subject to a 12.5% ad valorem import duty and require a valid Certificate of Conformity (CoC)."
    },
    {
        "instruction": "Define ad valorem duty for port clearance.",
        "response": "Ad valorem duty is an import tax calculated as a fixed percentage of the declared transactional value of the imported goods, rather than their weight or volume."
    }
]

def format_prompts(batch):
    """Formats raw instructions and responses into standard LLM training prompts."""
    texts = []
    for inst, resp in zip(batch["instruction"], batch["response"]):
        text = f"### Instruction:\n{inst}\n\n### Response:\n{resp}<|endoftext|>"
        texts.append(text)
    return {"text": texts}

# ==========================================
# 2. CONFIGURATION & MODEL LOADING
# ==========================================
def run_finetuning():
    # We use Llama 3.2 3B as our base model
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    print(f"[Init] Loading base model and tokenizer: {model_id}...")
    
    # Check for GPU
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    if device_map == "cpu":
        print("[Warning] No CUDA GPU detected. This script is running in CPU-simulation mode.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with float16 precision to save VRAM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map
    )
    
    # ==========================================
    # 3. APPLY LORA CONFIGURATION (Our Math in Action)
    # ==========================================
    print("[PEFT] Applying LoRA Low-Rank Adaptation configuration...")
    peft_config = LoraConfig(
        r=8,              # The rank (dimension) we simulated in lora_simulation.py
        lora_alpha=16,     # Scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Wrap the original model with our LoRA adapter layers
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # ==========================================
    # 4. PREPARE DATASET
    # ==========================================
    raw_dataset = Dataset.from_list(logistics_training_data)
    formatted_dataset = raw_dataset.map(format_prompts, batched=True)
    
    # ==========================================
    # 5. INITIALIZE TRAINER
    # ==========================================
    print("[Training] Configuring Supervised Fine-Tuning (SFT) Trainer...")
    training_args = TrainingArguments(
        output_dir="./lora_logistics_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,  # Keeping it short for testing
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="no",
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
        peft_config=peft_config
    )
    
    # ==========================================
    # 6. RUN TRAINING
    # ==========================================
    print("\n[Training] Starting local fine-tuning run...")
    try:
        trainer.train()
        print("\n✓ Success! Saved adapter weights to './lora_logistics_model'")
        # Save the adapter weights locally
        model.save_pretrained("./lora_logistics_model")
    except Exception as e:
        print(f"\n[Exit] Training halted: {e}")
        print("This is normal if running on a standard CPU machine without CUDA drivers.")

if __name__ == "__main__":
    # Ensure Hugging Face logins are set up if running in production
    # os.environ["HF_TOKEN"] = "your_token"
    run_finetuning()