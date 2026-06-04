#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Convertir LoRA a GGUF para Ollama")
    parser.add_argument("--base", required=True, help="Modelo base")
    parser.add_argument("--lora", required=True, help="Ruta de adaptadores LoRA")
    parser.add_argument("--output", required=True, help="Archivo GGUF de salida")
    parser.add_argument("--llama_cpp_dir", default="./llama.cpp", help="Directorio de llama.cpp")
    parser.add_argument("--merged_dir", default="./merged_model", help="Directorio temporal")
    return parser.parse_args()

def main():
    args = parse_args()
    convert_script = os.path.join(args.llama_cpp_dir, "convert_hf_to_gguf.py")
    
    if not os.path.isfile(convert_script):
        print(f"ERROR: No se encontró {convert_script}. Clona llama.cpp primero.")
        sys.exit(1)
        
    print(f"[1/4] Cargando tokenizer: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    
    print("[2/4] Cargando modelo base + LoRA y fusionando...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base_model, args.lora)
    model = model.merge_and_unload()
    
    os.makedirs(args.merged_dir, exist_ok=True)
    print(f"[3/4] Guardando modelo fusionado en: {args.merged_dir}")
    model.save_pretrained(args.merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_dir)
    
    del model
    del base_model
    
    print(f"[4/4] Convirtiendo a GGUF: {args.output}")
    result = subprocess.run(
        [sys.executable, convert_script, args.merged_dir, "--outfile", args.output, "--outtype", "f32"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"ERROR en conversión GGUF:\n{result.stderr}")
        sys.exit(1)
        
    print(result.stdout)
    print(f"\nConversión completada: {args.output}")

if __name__ == "__main__":
    main()