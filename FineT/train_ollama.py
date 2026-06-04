#!/usr/bin/env python3
"""
Entrenamiento LoRA para Qwen2.5-1.5B-Instruct — Tutor de Seguridad Pública.

Cambios respecto a la versión anterior:
  1. target_modules ampliado para incluir las capas MLP de Qwen2.5
     (gate_proj, up_proj, down_proj), que la versión anterior ignoraba.
  2. Formato de tokenización cambiado al chat template oficial de Qwen2.5
     (apply_chat_template) para que el modelo aprenda en el mismo formato
     en que luego infiere. Antes usaba "Instrucción:/Respuesta:" plano.
  3. Se añade --lora_modules para poder cambiar los módulos por CLI sin
     editar el código.
  4. bf16 activado automáticamente en Ampere+ (A100, RTX 30xx/40xx);
     es superior a fp16 para Qwen2.5.
  5. Se añade gradient_checkpointing para reducir uso de VRAM.
"""

import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Módulos LoRA correctos para Qwen2 / Qwen2.5
# - Atención: q_proj, k_proj, v_proj, o_proj
# - MLP:      gate_proj, up_proj, down_proj  ← Qwen2.5 los tiene; la versión
#             anterior los omitía, reduciendo la calidad del fine-tuning.
QWEN_LORA_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # atención
    "gate_proj", "up_proj", "down_proj",        # MLP (específico de Qwen2.x)
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenar LoRA sobre Qwen2.5 para el Tutor Analítico"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Modelo base de HuggingFace",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_seguridad.jsonl",
        help="Archivo JSONL con campos 'prompt' y 'completion'",
    )
    parser.add_argument("--output_dir",            type=str,   default="./lora-seguridad")
    parser.add_argument("--epochs",                type=int,   default=3)
    parser.add_argument("--max_length",            type=int,   default=512)
    parser.add_argument("--batch_size",            type=int,   default=1)
    parser.add_argument("--gradient_accumulation", type=int,   default=16)
    parser.add_argument("--learning_rate",         type=float, default=2e-4)
    parser.add_argument("--lora_r",                type=int,   default=16)
    parser.add_argument("--lora_alpha",            type=int,   default=32)
    parser.add_argument(
        "--lora_modules",
        type=str,
        default=",".join(QWEN_LORA_MODULES),
        help="Módulos LoRA separados por coma (default: todos los de Qwen2.5)",
    )
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="No usar cuantización 8-bit (necesario en CPU)",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, use_quantization: bool):
    """Carga modelo base y tokenizer de Qwen2.5."""
    print(f"Cargando tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Qwen2.5 define su propio pad_token; si no viene, usamos eos.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    has_cuda = torch.cuda.is_available()

    # Detectar soporte bf16 (Ampere+: RTX 30xx, A100, etc.)
    supports_bf16 = has_cuda and torch.cuda.is_bf16_supported()

    if use_quantization and has_cuda:
        print("Cargando modelo en 8-bit (cuantización habilitada)")
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,   # ahorra VRAM
        )
    else:
        if not has_cuda:
            print("AVISO: No se detectó GPU. Entrenando en CPU (lento).")
        dtype = torch.bfloat16 if supports_bf16 else (
                torch.float16  if has_cuda else torch.float32)
        print(f"Cargando modelo con dtype={dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if has_cuda else None,
            trust_remote_code=True,
        )
        if has_cuda:
            model.gradient_checkpointing_enable()

    model.config.use_cache = False
    return model, tokenizer


def apply_lora(model, lora_r: int, lora_alpha: int, modules: list[str]):
    """Aplica adaptadores LoRA al modelo con los módulos correctos para Qwen2.5."""
    print(f"Aplicando LoRA a módulos: {modules}")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=modules,        # ← ampliado vs versión anterior
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def prepare_dataset(dataset_path: str, tokenizer, max_length: int):
    """
    Carga y tokeniza el dataset usando el chat template oficial de Qwen2.5.

    Antes se usaba "Instrucción: X\nRespuesta: Y", que no coincide con el
    formato en que Qwen2.5-Instruct fue pre-entrenado. Usar apply_chat_template
    garantiza que el fine-tuning ocurra en el mismo espacio de tokens.
    """
    print(f"Cargando dataset: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path)

    def tokenize(example):
        # Soporta dos formatos:
        #   - Nuevo (dataset_seguridad_chat.jsonl): campo 'messages' con roles
        #   - Legado (dataset_seguridad.jsonl):     campos 'prompt'/'completion'
        if "messages" in example:
            messages = example["messages"]
        else:
            messages = [
                {"role": "user",      "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]},
            ]
        # apply_chat_template agrega <|im_start|> / <|im_end|> correctamente
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset["train"].column_names,
    )
    print(f"Dataset tokenizado: {len(tokenized['train'])} ejemplos")
    return tokenized


def main():
    args = parse_args()
    modules = [m.strip() for m in args.lora_modules.split(",")]

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        use_quantization=not args.no_quantize,
    )
    model = apply_lora(model, args.lora_r, args.lora_alpha, modules)
    tokenized = prepare_dataset(args.dataset, tokenizer, args.max_length)

    has_cuda = torch.cuda.is_available()
    supports_bf16 = has_cuda and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        # bf16 > fp16 en hardware Ampere+; fp16 como fallback
        bf16=supports_bf16,
        fp16=has_cuda and not supports_bf16,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,   # reduce VRAM ~30%
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=data_collator,
    )

    print("\n--- Iniciando entrenamiento ---")
    trainer.train()
    print(f"\nGuardando adaptadores LoRA en: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Entrenamiento completado.")


if __name__ == "__main__":
    main()