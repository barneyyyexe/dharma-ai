from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("data/processed/adi_parva_chunks.jsonl")
OUTPUT_FILE = Path("data/processed/adi_parva_embeddings.npz")

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading Mahabharata chunks...")

    chunks = []

    with INPUT_FILE.open(
        encoding="utf-8"
    ) as file:

        for line in file:
            chunks.append(json.loads(line))

    print(f"Chunks loaded: {len(chunks)}")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print("\nEmbedding matrix shape:")
    print(embeddings.shape)

    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=embeddings,
    )

    print("\nEmbeddings saved.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Vectors: {len(embeddings)}")
    print(f"Dimensions: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()