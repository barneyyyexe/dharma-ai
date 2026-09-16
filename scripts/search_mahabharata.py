from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/adi_parva_chunks.jsonl")
EMBEDDINGS_FILE = Path("data/processed/adi_parva_embeddings.npz")

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


def load_chunks():
    chunks = []

    with CHUNKS_FILE.open(
        encoding="utf-8"
    ) as file:

        for line in file:
            chunks.append(json.loads(line))

    return chunks


def load_embeddings():
    data = np.load(EMBEDDINGS_FILE)

    return data["embeddings"]


def search(query, model, chunks, embeddings):
    # Convert the user's question into the same
    # 384-dimensional embedding space.
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    # Because both vectors are normalized,
    # dot product = cosine similarity.
    scores = embeddings @ query_embedding

    # Get the indexes of the highest-scoring chunks.
    top_indexes = np.argsort(scores)[::-1][:TOP_K]

    results = []

    for index in top_indexes:

        results.append(
            {
                "score": float(scores[index]),
                "chunk": chunks[index],
            }
        )

    return results


def main():

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}"
        )

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading chunks...")
    chunks = load_chunks()

    print("Loading embeddings...")
    embeddings = load_embeddings()

    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {embeddings.shape}")

    print("\n" + "=" * 70)
    print("MAHABHARATA SEMANTIC SEARCH")
    print("=" * 70)

    while True:

        query = input(
            "\nEnter your dilemma "
            "(or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            print("Goodbye.")
            break

        if not query:
            continue

        results = search(
            query,
            model,
            chunks,
            embeddings,
        )

        print("\nTop results:\n")

        for rank, result in enumerate(results, 1):

            chunk = result["chunk"]

            print("=" * 70)
            print(
                f"#{rank} | "
                f"Similarity: {result['score']:.4f}"
            )

            print(
                f"Source: "
                f"{chunk['parva']}, "
                f"Section {chunk['section_number']}"
            )

            print(
                f"Chunk: {chunk['chunk_number']}"
            )

            print("-" * 70)

            print(chunk["text"])

            print()


if __name__ == "__main__":
    main()