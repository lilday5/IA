import json
from pathlib import Path

import numpy as np
import tensorflow as tf

# ── Cargar modelo y metadatos ─────────────────────────────────
ROOT = Path("rnn-keras-autocomplete")

if not (ROOT / "model.keras").exists():
    print("ERROR: No existe model.keras — corre train.py primero.")
    exit(1)

meta = json.loads((ROOT / "meta.json").read_text(encoding="utf-8"))
BLOCK_SIZE = meta["block_size"]
chars = meta["chars"]
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

model = tf.keras.models.load_model(ROOT / "model.keras")
print("Modelo cargado. Vocabulario:", len(chars), "caracteres\n")


# ── Helpers ───────────────────────────────────────────────────
def encode(text):
    # ignora caracteres fuera del vocabulario
    return [stoi[c] for c in text if c in stoi]


def decode(ids):
    return "".join(itos[i] for i in ids)


def complete(prompt, max_new=120, temperature=0.75):
    ids = encode(prompt)
    if not ids:
        return prompt  # prefijo vacío o sin caracteres conocidos

    rng = np.random.default_rng()

    for _ in range(max_new):
        x = np.array(ids[-BLOCK_SIZE:], dtype=np.int64)

        # padding si el contexto es más corto que BLOCK_SIZE
        if len(x) < BLOCK_SIZE:
            pad = np.zeros(BLOCK_SIZE - len(x), dtype=np.int64)
            x = np.concatenate([pad, x])

        logits = model(x.reshape(1, BLOCK_SIZE), training=False).numpy()[0, -1]
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()

        ids.append(int(rng.choice(len(probs), p=probs)))

    return decode(ids)


# ── Loop interactivo ──────────────────────────────────────────
print("Escribe un prefijo de código C y Enter para autocompletar.")
print("(Ctrl+C para salir)\n")

while True:
    try:
        texto = input("Prefijo: ")
        if not texto.strip():
            continue
        resultado = complete(texto)
        print("\n─── Resultado ───")
        print(resultado)
        print("─────────────────\n")
    except KeyboardInterrupt:
        print("\nSaliendo.")
        break