"""LoRA fine-tuning of a small base model to explain quant signals.

Builds a synthetic dataset from the deterministic explainer, then fine-tunes a
small instruction model with LoRA/PEFT. Defaults are CPU-friendly for a smoke
run; use a GPU and a larger base model for real training.

Requires the optional extra::

    uv sync --extra finetune
    uv run python scripts/finetune_lora.py --max-samples 64 --epochs 1
"""

from __future__ import annotations

import argparse
import random

from quantlens.explain import rule_based

_TICKERS = ("PETR4", "VALE3", "ITUB4", "BBAS3", "ABEV3", "BBDC4", "B3SA3", "WEGE3")


def build_dataset(n: int, seed: int = 0) -> list[dict[str, str]]:
    """Synthesize (prompt, completion) pairs from the rule-based explainer."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for _ in range(n):
        ticker = rng.choice(_TICKERS)
        rsi = rng.uniform(10, 90)
        mom = rng.uniform(-0.15, 0.15)
        vol = rng.uniform(0.15, 0.60)
        prompt = (
            f"Explain the signals for {ticker}: RSI {rsi:.0f}, "
            f"momentum {mom:+.2%}, annualized volatility {vol:.2%}."
        )
        rows.append({"prompt": prompt, "completion": rule_based(ticker, rsi, mom, vol)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune a signal explainer.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="models/explainer-lora")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Heavy deps are imported lazily so the rest of the project never needs them.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    rows = build_dataset(args.max_samples, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(row: dict[str, str]) -> dict[str, str]:
        body = f"<|user|>\n{row['prompt']}\n<|assistant|>\n{row['completion']}"
        return {"text": body + tokenizer.eos_token}

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(batch["text"], truncation=True, max_length=256)

    dataset = Dataset.from_list([to_text(row) for row in rows]).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM"),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=4,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="epoch",
            report_to=[],
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
