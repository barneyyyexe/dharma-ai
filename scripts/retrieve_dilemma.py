from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Files
# ---------------------------------------------------------

CHUNKS_FILE = Path("data/processed/adi_parva_chunks.jsonl")
EMBEDDINGS_FILE = Path("data/processed/adi_parva_embeddings.npz")
WISDOM_FILE = Path("data/knowledge/wisdom_map.jsonl")

MODEL_NAME = "all-MiniLM-L6-v2"

SEMANTIC_TOP_K = 5
WISDOM_TOP_K = 3


# ---------------------------------------------------------
# Load Mahabharata chunks
# ---------------------------------------------------------

def load_chunks():
    chunks = []

    with CHUNKS_FILE.open(encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


# ---------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------

def load_embeddings():
    data = np.load(EMBEDDINGS_FILE)
    return data["embeddings"]


# ---------------------------------------------------------
# Load Wisdom Map
# ---------------------------------------------------------

def load_wisdom():
    records = []

    with WISDOM_FILE.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


# ---------------------------------------------------------
# Semantic search
# ---------------------------------------------------------

def semantic_search(query, model, chunks, embeddings):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    scores = embeddings @ query_embedding

    top_indexes = np.argsort(scores)[::-1][:SEMANTIC_TOP_K]

    results = []

    for index in top_indexes:
        results.append({
            "score": float(scores[index]),
            "chunk": chunks[index],
        })

    return results


# ---------------------------------------------------------
# Wisdom search
# ---------------------------------------------------------

def wisdom_search(query, wisdom_records):
    query_words = set(query.lower().split())

    results = []

    for record in wisdom_records:

        searchable_text = " ".join([
            record["situation"],
            record["lesson"],
            record["interpretation"],
            record["caution"],
            " ".join(record["dilemma_types"]),
            " ".join(record["values_in_conflict"]),
            " ".join(record["emotions"]),
            " ".join(record["relevance_to_modern_life"]),
            " ".join(record["keywords"]),
        ]).lower()

        score = sum(
            1
            for word in query_words
            if word in searchable_text
        )

        if score > 0:
            results.append({
                "score": score,
                "record": record,
            })

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:WISDOM_TOP_K]


# ---------------------------------------------------------
# Combined retrieval
# ---------------------------------------------------------

def retrieve_dilemma(
    query,
    model,
    chunks,
    embeddings,
    wisdom_records,
):

    semantic_results = semantic_search(
        query,
        model,
        chunks,
        embeddings,
    )

    wisdom_results = wisdom_search(
        query,
        wisdom_records,
    )

    return {
        "query": query,
        "semantic_results": semantic_results,
        "wisdom_results": wisdom_results,
    }


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

def display_results(results):

    print("\n")
    print("=" * 70)
    print("DILEMMA RETRIEVAL")
    print("=" * 70)

    print("\nUSER DILEMMA:")
    print(results["query"])

    # -----------------------------------------------------
    # Mahabharata results
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RELEVANT MAHABHARATA PASSAGES")
    print("=" * 70)

    for rank, result in enumerate(
        results["semantic_results"],
        1,
    ):

        chunk = result["chunk"]

        print("\n" + "-" * 70)
        print(
            f"#{rank} | Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"Source: {chunk['parva']}, "
            f"Section {chunk['section_number']}, "
            f"Chunk {chunk['chunk_number']}"
        )

        print("\n")
        print(chunk["text"])

    # -----------------------------------------------------
    # Wisdom results
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RELEVANT WISDOM")
    print("=" * 70)

    if not results["wisdom_results"]:
        print("\nNo wisdom records matched.")

    for rank, result in enumerate(
        results["wisdom_results"],
        1,
    ):

        record = result["record"]

        print("\n" + "-" * 70)

        print(
            f"#{rank} | Keyword score: "
            f"{result['score']}"
        )

        print(f"Source: {record['source']}")

        print(
            f"Characters: "
            f"{', '.join(record['characters'])}"
        )

        print(
            f"Dilemma types: "
            f"{', '.join(record['dilemma_types'])}"
        )

        print("\nLesson:")
        print(record["lesson"])

        print("\nInterpretation:")
        print(record["interpretation"])

        print("\nCaution:")
        print(record["caution"])


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Loading Mahabharata chunks...")

    chunks = load_chunks()

    print("Loading embeddings...")

    embeddings = load_embeddings()

    print("Loading Wisdom Map...")

    wisdom_records = load_wisdom()

    print()
    print("=" * 70)
    print("DILEMMA RETRIEVAL ENGINE")
    print("=" * 70)

    print(f"Mahabharata chunks: {len(chunks)}")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Wisdom records: {len(wisdom_records)}")

    while True:

        query = input(
            "\nEnter your dilemma (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            print("Goodbye.")
            break

        if not query:
            continue

        results = retrieve_dilemma(
            query,
            model,
            chunks,
            embeddings,
            wisdom_records,
        )

        display_results(results)


if __name__ == "__main__":
    main()