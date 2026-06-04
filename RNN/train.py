import json
from pathlib import Path

import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

# ── Leer dataset ──────────────────────────────────────────────
with open("dataset_c.txt", "r", encoding="utf-8") as f:
    CORPUS = f.read()

print(f"Corpus cargado: {len(CORPUS)} caracteres")

# ── Vocabulario ───────────────────────────────────────────────
chars = sorted(set(CORPUS))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(chars)

print(f"VOCAB_SIZE: {VOCAB_SIZE}")


def encode(s):
    return [stoi[c] for c in s]


SEQ = np.array(encode(CORPUS), dtype=np.int64)

# ── Ventanas de entrenamiento ─────────────────────────────────
BLOCK_SIZE = 32
X_rows, Y_rows = [], []

for i in range(len(SEQ) - BLOCK_SIZE):
    X_rows.append(SEQ[i : i + BLOCK_SIZE])
    Y_rows.append(SEQ[i + 1 : i + 1 + BLOCK_SIZE])

X = np.stack(X_rows)
Y = np.stack(Y_rows)
print(f"X: {X.shape}  Y: {Y.shape}")

# ── Modelo ────────────────────────────────────────────────────
EMBED_DIM = 48
HIDDEN = 64

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(BLOCK_SIZE,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM),
    tf.keras.layers.SimpleRNN(HIDDEN, activation="tanh", return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(VOCAB_SIZE)),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

model.summary()

# ── Entrenamiento ─────────────────────────────────────────────
EPOCHS = 120
BATCH_SIZE = 16

history = model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)

print(f"\nPérdida inicial : {history.history['loss'][0]:.4f}")
print(f"Pérdida final   : {history.history['loss'][-1]:.4f}")

# ── Guardar modelo y metadatos ────────────────────────────────
DEPLOY_DIR = Path("rnn-keras-autocomplete")
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

model.save(DEPLOY_DIR / "model.keras")

(DEPLOY_DIR / "meta.json").write_text(
    json.dumps({"block_size": BLOCK_SIZE, "chars": chars}, ensure_ascii=False),
    encoding="utf-8",
)

print(f"\nGuardado en: {DEPLOY_DIR.resolve()}")