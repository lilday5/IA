#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

SYSTEM_PROMPT = """Eres un Tutor Analítico especializado en seguridad pública y violencia en México.
Sigues estas reglas de forma estricta e inquebrantable:
1. TONO ACADÉMICO: Mantén absoluta neutralidad y objetividad ante temas sensibles. Cero sesgos ideológicos. Lenguaje formal y preciso.
2. CITAR FUENTES: Al mencionar cualquier dato estadístico o afirmación factual, incluye obligatoriamente la referencia al final en este formato: [Fuente: Nombre del documento, Pág. X] o [cite: N].
3. MÉTODO SOCRÁTICO: Guía con preguntas reflexivas. Formula UNA pregunta al final de tu respuesta.
4. MANEJO DE INCERTIDUMBRE: Si la información no está en el corpus, responde EXACTAMENTE: "La información proporcionada en el corpus no detalla este aspecto."
5. ESTRUCTURA: Respuestas bien organizadas. Máximo 4 párrafos."""

PARAMS = """PARAMETER num_ctx 4096
PARAMETER temperature 0.3
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "[Fragmento"
PARAMETER stop "Pregunta:"
PARAMETER stop "Contexto:"
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_seguridad.jsonl")
    parser.add_argument("--base", default="qwen2.5:1.5b")
    parser.add_argument("--output", default="Modelfile")
    parser.add_argument("--max_examples", type=int, default=20)
    return parser.parse_args()

def load_examples(path, max_n):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj:
                msgs = obj["messages"]
            else:
                msgs = [{"role": "user", "content": obj["prompt"]},
                        {"role": "assistant", "content": obj["completion"]}]
            examples.append(msgs)
            if len(examples) >= max_n:
                break
    return examples

def build_modelfile(base, system, params, examples):
    lines = [f"FROM {base}", "", params.strip(), "",
             'SYSTEM """' + system + '"""', ""]
    for msgs in examples:
        for msg in msgs:
            content = msg["content"].replace('"""', "'''")
            lines.append(f'MESSAGE {msg["role"]} """{content}"""')
        lines.append("")
    return "\n".join(lines)

def main():
    args = parse_args()
    if not Path(args.dataset).exists():
        print(f"ERROR: No se encontró {args.dataset}")
        return
    examples = load_examples(args.dataset, args.max_examples)
    print(f"  → {len(examples)} pares cargados")
    content = build_modelfile(args.base, SYSTEM_PROMPT, PARAMS, examples)
    Path(args.output).write_text(content, encoding="utf-8")
    print(f"Modelfile escrito. Ahora ejecuta:")
    print(f"  ollama create tutor_analitico -f {args.output}")

if __name__ == "__main__":
    main()
