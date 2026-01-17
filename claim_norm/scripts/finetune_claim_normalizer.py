#!/usr/bin/env python3
"""
Fine-tune a language model for claim normalization using QLoRA.

GPU Presets:
- RTX 5090 (32GB): Best for 7B-14B models, can run 7B without quantization
- GH200 (96GB): Can fine-tune 70B+ models with 4-bit quantization
- A10/A100 (24-40GB): Good for 7B-14B models with 4-bit

Usage:
    # Auto-detect GPU and use optimal settings
    python finetune_claim_normalizer.py --model mistral-7b --epochs 3

    # RTX 5090 optimized (32GB VRAM)
    python finetune_claim_normalizer.py --model qwen-14b --gpu rtx5090 --epochs 3

    # GH200 optimized (96GB VRAM) - can run larger models
    python finetune_claim_normalizer.py --model qwen-72b --gpu gh200 --epochs 3

    # Custom settings
    python finetune_claim_normalizer.py --model mistral-7b --batch-size 8 --lr 1e-4

    # Resume from checkpoint
    python finetune_claim_normalizer.py --model mistral-7b --resume checkpoints/latest
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# =============================================================================
# GPU Presets - Optimized configurations for different hardware
# =============================================================================

@dataclass
class GPUPreset:
    """Configuration preset for a specific GPU."""
    name: str
    vram_gb: int
    batch_size: int
    gradient_accumulation: int
    use_4bit: bool
    use_8bit: bool
    use_bf16: bool  # True for modern GPUs (Ampere+, Blackwell)
    use_flash_attn: bool
    lora_r: int  # LoRA rank - higher = better quality, more VRAM
    lora_alpha: int
    max_model_size: str  # Recommended max model size
    torch_compile: bool  # Use torch.compile for speedup


GPU_PRESETS = {
    "rtx5090": GPUPreset(
        name="RTX 5090",
        vram_gb=32,
        batch_size=4,  # Conservative for 14B models
        gradient_accumulation=4,
        use_4bit=True,  # Required for 14B models
        use_8bit=False,
        use_bf16=True,  # Blackwell has excellent bf16
        use_flash_attn=True,
        lora_r=32,  # Higher rank for better quality
        lora_alpha=64,
        max_model_size="14B",
        torch_compile=False,  # Disable to save memory during first pass
    ),
    "rtx5090_7b": GPUPreset(
        name="RTX 5090 (7B no quant)",
        vram_gb=32,
        batch_size=8,  # Higher batch for smaller model
        gradient_accumulation=2,
        use_4bit=False,  # 7B can run without quantization
        use_8bit=False,
        use_bf16=True,
        use_flash_attn=True,
        lora_r=32,
        lora_alpha=64,
        max_model_size="7B",
        torch_compile=True,  # Can use compile with 7B
    ),
    "gh200": GPUPreset(
        name="GH200 Grace Hopper",
        vram_gb=96,
        batch_size=4,
        gradient_accumulation=4,
        use_4bit=True,  # For 70B models
        use_8bit=False,
        use_bf16=True,
        use_flash_attn=True,
        lora_r=64,  # Can afford higher rank
        lora_alpha=128,
        max_model_size="72B",
        torch_compile=False,  # May have issues on GH200
    ),
    "a100_40gb": GPUPreset(
        name="A100 40GB",
        vram_gb=40,
        batch_size=4,
        gradient_accumulation=4,
        use_4bit=True,
        use_8bit=False,
        use_bf16=True,
        use_flash_attn=True,
        lora_r=32,
        lora_alpha=64,
        max_model_size="14B",
        torch_compile=False,
    ),
    "a10_24gb": GPUPreset(
        name="A10 24GB",
        vram_gb=24,
        batch_size=4,
        gradient_accumulation=4,
        use_4bit=True,
        use_8bit=False,
        use_bf16=False,  # A10 bf16 can be slow
        use_flash_attn=True,
        lora_r=16,
        lora_alpha=32,
        max_model_size="7B",
        torch_compile=False,
    ),
    "rtx4090": GPUPreset(
        name="RTX 4090",
        vram_gb=24,
        batch_size=4,
        gradient_accumulation=4,
        use_4bit=True,
        use_8bit=False,
        use_bf16=True,
        use_flash_attn=True,
        lora_r=16,
        lora_alpha=32,
        max_model_size="7B",
        torch_compile=True,
    ),
    "default": GPUPreset(
        name="Default (conservative)",
        vram_gb=16,
        batch_size=2,
        gradient_accumulation=8,
        use_4bit=True,
        use_8bit=False,
        use_bf16=False,
        use_flash_attn=False,
        lora_r=16,
        lora_alpha=32,
        max_model_size="7B",
        torch_compile=False,
    ),
}


def detect_gpu() -> str:
    """Auto-detect GPU and return best preset name."""
    if not torch.cuda.is_available():
        return "default"

    gpu_name = torch.cuda.get_device_name(0).lower()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"Detected GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB)")

    # Match by name
    if "5090" in gpu_name:
        return "rtx5090"
    elif "gh200" in gpu_name or "grace" in gpu_name:
        return "gh200"
    elif "a100" in gpu_name:
        return "a100_40gb" if vram_gb > 45 else "a10_24gb"
    elif "4090" in gpu_name:
        return "rtx4090"
    elif "a10" in gpu_name:
        return "a10_24gb"
    elif vram_gb >= 90:
        return "gh200"
    elif vram_gb >= 30:
        return "rtx5090"
    elif vram_gb >= 20:
        return "rtx4090"
    else:
        return "default"


# =============================================================================
# Model configurations
# =============================================================================

MODELS = {
    # 7B models - fit on most GPUs
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    # 14B models - good for RTX 5090
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    # Large models - require 4-bit on most GPUs
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
}

# Model sizes in billions of parameters (for VRAM estimation)
MODEL_SIZES = {
    "mistral-7b": 7,
    "qwen-7b": 7,
    "llama-8b": 8,
    "qwen-14b": 14,
    "mixtral-8x7b": 47,  # MoE, but ~12B active
    "qwen-72b": 72,
    "llama-70b": 70,
}

# System prompt for claim normalization
SYSTEM_PROMPT = """You are a claim normalization specialist. Transform social media posts into clean, factual claims for fact-checking.

INSTRUCTIONS:
1. Extract the main factual assertion from the post
2. Rewrite it as a single, clear declarative sentence
3. Keep all specific details: names, numbers, dates, locations
4. Remove: emojis, hashtags, @mentions, URLs, "See More", repeated text
5. Use third person and neutral tone

Output ONLY the normalized claim - no explanations, no quotes, no prefixes."""


# =============================================================================
# Data loading and preprocessing
# =============================================================================

def load_training_data(data_dir: Path) -> list[dict]:
    """Load training data from CT25 CSV files."""
    import polars as pl

    train_path = data_dir / "train-eng.csv"
    dev_path = data_dir / "dev-eng.csv"

    datasets = []

    for path in [train_path, dev_path]:
        if path.exists():
            df = pl.read_csv(path)
            if "normalized claim" in df.columns:
                df = df.rename({"normalized claim": "gold_claim"})

            # Filter to rows with gold claims
            df = df.filter(pl.col("gold_claim").is_not_null())

            for row in df.iter_rows(named=True):
                datasets.append({
                    "post": row["post"],
                    "claim": row["gold_claim"],
                })

    print(f"Loaded {len(datasets)} training examples")
    return datasets


def format_training_example(example: dict, tokenizer) -> dict:
    """Format a single example for training."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Normalize this post:\n\n{example['post'][:2000]}"},
        {"role": "assistant", "content": example["claim"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def create_dataset(examples: list[dict], tokenizer, max_length: int = 2048) -> Dataset:
    """Create HuggingFace dataset from examples."""
    formatted = [format_training_example(ex, tokenizer) for ex in examples]
    dataset = Dataset.from_list(formatted)

    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(tokenize, remove_columns=["text"])
    return dataset


# =============================================================================
# Model loading with GPU-specific optimizations
# =============================================================================

def load_model_for_training(model_name: str, preset: GPUPreset):
    """Load model with GPU-optimized QLoRA configuration."""

    print(f"\nLoading model: {model_name}")
    print(f"GPU preset: {preset.name}")
    print(f"Quantization: {'4-bit' if preset.use_4bit else '8-bit' if preset.use_8bit else 'none'}")
    print(f"Precision: {'bf16' if preset.use_bf16 else 'fp16'}")

    # Determine compute dtype
    compute_dtype = torch.bfloat16 if preset.use_bf16 else torch.float16

    # Quantization config
    if preset.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif preset.use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine attention implementation
    if preset.use_flash_attn:
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            attn_impl = "sdpa"
            print("Flash Attention not available, using SDPA")
    else:
        attn_impl = "sdpa"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        attn_implementation=attn_impl,
    )

    # Prepare for k-bit training
    if preset.use_4bit or preset.use_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Enable input gradients (required for PEFT)
    model.enable_input_require_grads()

    # LoRA configuration with preset values
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=preset.lora_r,
        lora_alpha=preset.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",  # MLP
        ],
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Optional: torch.compile for Blackwell/Ampere+
    if preset.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile failed (non-fatal): {e}")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"LoRA rank: {preset.lora_r}, alpha: {preset.lora_alpha}")

    return model, tokenizer


# =============================================================================
# Training
# =============================================================================

def train(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    preset: GPUPreset,
    epochs: int = 3,
    batch_size: int | None = None,
    gradient_accumulation: int | None = None,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
    warmup_ratio: float = 0.1,
    save_steps: int = 100,
    eval_steps: int = 100,
    resume_from: str | None = None,
):
    """Main training function with GPU-optimized settings."""

    # Use preset defaults if not overridden
    batch_size = batch_size or preset.batch_size
    gradient_accumulation = gradient_accumulation or preset.gradient_accumulation

    # Load model and tokenizer
    model, tokenizer = load_model_for_training(model_name, preset)

    # Load and prepare data
    examples = load_training_data(data_dir)

    # Split into train/eval (90/10)
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]

    print(f"Train examples: {len(train_examples)}")
    print(f"Eval examples: {len(eval_examples)}")

    # Create datasets
    train_dataset = create_dataset(train_examples, tokenizer, max_length)
    eval_dataset = create_dataset(eval_examples, tokenizer, max_length)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Training arguments with GPU-specific optimizations
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=False,  # Use last checkpoint (avoids eval_loss issue)
        bf16=preset.use_bf16,
        fp16=not preset.use_bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        # RTX 5090 / Blackwell optimizations
        tf32=preset.use_bf16,  # Enable TF32 on Ampere+
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Resume from checkpoint if specified
    if resume_from:
        print(f"Resuming from: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save the final model
    final_path = output_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save training config
    config = {
        "base_model": model_name,
        "gpu_preset": preset.name,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "learning_rate": learning_rate,
        "lora_r": preset.lora_r,
        "lora_alpha": preset.lora_alpha,
        "use_4bit": preset.use_4bit,
        "use_bf16": preset.use_bf16,
        "timestamp": datetime.now().isoformat(),
    }
    with open(final_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete! Model saved to: {final_path}")

    return final_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune model for claim normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GPU Presets:
  rtx5090      RTX 5090 32GB - Best for 7B-14B, high batch size, bf16
  rtx5090_4bit RTX 5090 32GB with 4-bit - For larger models
  gh200        GH200 96GB - Can fine-tune 70B+ models
  a100_40gb    A100 40GB - Good for 14B models
  a10_24gb     A10 24GB - 7B models with 4-bit
  rtx4090      RTX 4090 24GB - Similar to A10
  default      Conservative settings for unknown GPUs

Examples:
  # RTX 5090 with Qwen-14B (optimal for 32GB)
  python finetune_claim_normalizer.py --model qwen-14b --gpu rtx5090

  # RTX 5090 with Mistral-7B (no quantization needed)
  python finetune_claim_normalizer.py --model mistral-7b --gpu rtx5090

  # GH200 with Qwen-72B
  python finetune_claim_normalizer.py --model qwen-72b --gpu gh200
        """
    )
    parser.add_argument("--model", type=str, default="mistral-7b",
                        choices=list(MODELS.keys()),
                        help="Base model to fine-tune")
    parser.add_argument("--gpu", type=str, default=None,
                        choices=list(GPU_PRESETS.keys()),
                        help="GPU preset (auto-detected if not specified)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing training data")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (uses GPU preset default if not specified)")
    parser.add_argument("--gradient-accumulation", type=int, default=None,
                        help="Gradient accumulation steps (uses preset default)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    args = parser.parse_args()

    # Auto-detect GPU if not specified
    gpu_preset_name = args.gpu or detect_gpu()
    preset = GPU_PRESETS[gpu_preset_name]

    # Check if model fits on GPU
    model_size = MODEL_SIZES.get(args.model, 7)
    max_model_size = int(preset.max_model_size.replace("B", ""))

    if model_size > max_model_size and not preset.use_4bit:
        print(f"\n⚠ Warning: {args.model} ({model_size}B) may not fit on {preset.name}")
        print(f"  Recommended max: {preset.max_model_size}")
        print(f"  Consider using --gpu {gpu_preset_name}_4bit or a smaller model\n")

    # Resolve paths
    model_name = MODELS[args.model]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Effective batch size
    batch_size = args.batch_size or preset.batch_size
    grad_accum = args.gradient_accumulation or preset.gradient_accumulation

    print("=" * 60)
    print("CLAIM NORMALIZER FINE-TUNING")
    print("=" * 60)
    print(f"GPU Preset:    {preset.name} ({preset.vram_gb}GB)")
    print(f"Base model:    {model_name}")
    print(f"Model size:    {model_size}B parameters")
    print(f"Data dir:      {data_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    print(f"Learning rate: {args.lr}")
    print(f"LoRA rank:     {preset.lora_r}")
    print(f"Quantization:  {'4-bit' if preset.use_4bit else '8-bit' if preset.use_8bit else 'none (full precision)'}")
    print(f"Precision:     {'bf16' if preset.use_bf16 else 'fp16'}")
    print("=" * 60)

    # Check for required files
    if not (data_dir / "train-eng.csv").exists():
        print(f"\nError: Training data not found at {data_dir / 'train-eng.csv'}")
        print("Please ensure the CheckThat! 2025 data files are in the data directory.")
        return

    # Train
    final_path = train(
        model_name=model_name,
        data_dir=data_dir,
        output_dir=output_dir,
        preset=preset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.lr,
        max_length=args.max_length,
        resume_from=args.resume,
    )

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"1. Run inference with the fine-tuned model:")
    print(f"   python run_claim_normalization_ct25.py --local --local-model {final_path} --split test")
    print()
    print(f"2. Or use in the existing pipeline by updating LOCAL_MODELS")
    print("=" * 60)


if __name__ == "__main__":
    main()
