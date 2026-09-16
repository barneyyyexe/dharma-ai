from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from dilemma_analyzer import analyze_dilemma
from app.reasoning.engine import reason
from app.ai.gemini import GeminiClient
from app.ai.prompt_builder import build_prompt


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
# Build enriched retrieval query
# ---------------------------------------------------------

def build_retrieval_query(dilemma):
    parts = [
        dilemma.original_text,
        " ".join(dilemma.emotions),
        " ".join(dilemma.situation),
        " ".join(dilemma.desired_actions),
        " ".join(dilemma.underlying_conflicts),
        " ".join(dilemma.values),
    ]

    return " ".join(
        part
        for part in parts
        if part
    )


# ---------------------------------------------------------
# Semantic search
# ---------------------------------------------------------

def semantic_search(
    query,
    model,
    chunks,
    embeddings,
):
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

def wisdom_search(
    dilemma,
    wisdom_records,
):
    query_terms = set()

    query_terms.update(dilemma.emotions)
    query_terms.update(dilemma.situation)
    query_terms.update(dilemma.desired_actions)
    query_terms.update(dilemma.underlying_conflicts)
    query_terms.update(dilemma.values)

    results = []

    for record in wisdom_records:

        searchable_fields = [
            record["situation"],
            record["lesson"],
            record["interpretation"],
            record["caution"],
        ]

        searchable_fields.extend(
            record["dilemma_types"]
        )

        searchable_fields.extend(
            record["values_in_conflict"]
        )

        searchable_fields.extend(
            record["emotions"]
        )

        searchable_fields.extend(
            record["relevance_to_modern_life"]
        )

        searchable_fields.extend(
            record["keywords"]
        )

        searchable_text = " ".join(
            searchable_fields
        ).lower()

        matched_terms = []

        for term in query_terms:
            if term.lower() in searchable_text:
                matched_terms.append(term)

        score = len(matched_terms)

        if score > 0:
            results.append({
                "score": score,
                "matched_terms": matched_terms,
                "record": record,
            })

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:WISDOM_TOP_K]


# ---------------------------------------------------------
# Combined retrieval + reranking
# ---------------------------------------------------------

def retrieve_dilemma(
    dilemma,
    model,
    chunks,
    embeddings,
    wisdom_records,
):
    retrieval_query = build_retrieval_query(
        dilemma
    )

    # -----------------------------------------------------
    # Step 1 — Semantic retrieval
    # -----------------------------------------------------

    semantic_results = semantic_search(
        retrieval_query,
        model,
        chunks,
        embeddings,
    )

    # -----------------------------------------------------
    # Step 2 — Wisdom retrieval
    # -----------------------------------------------------

    wisdom_results = wisdom_search(
        dilemma,
        wisdom_records,
    )

    # -----------------------------------------------------
    # Step 3 — Collect sources represented in Wisdom Map
    # -----------------------------------------------------

    wisdom_sources = set()

    for result in wisdom_results:
        record = result["record"]

        source = record.get("source", "").strip().lower()

        if source:
            wisdom_sources.add(source)

    # -----------------------------------------------------
    # Step 4 — Rerank semantic results
    # -----------------------------------------------------

    reranked_results = []

    for result in semantic_results:

        chunk = result["chunk"]

        semantic_score = result["score"]

        chunk_source = (
            f"{chunk.get('parva', '')}, "
            f"Section {chunk.get('section', '')}"
        ).lower()

        wisdom_boost = 0.0

        for wisdom_source in wisdom_sources:
            if wisdom_source in chunk_source:
                wisdom_boost = 0.15
                break

        final_score = semantic_score + wisdom_boost

        reranked_results.append({
            "score": semantic_score,
            "wisdom_boost": wisdom_boost,
            "final_score": final_score,
            "chunk": chunk,
        })

    # -----------------------------------------------------
    # Step 5 — Sort by final score
    # -----------------------------------------------------

    reranked_results.sort(
        key=lambda result: result["final_score"],
        reverse=True,
    )

    return {
        "dilemma": dilemma,
        "retrieval_query": retrieval_query,
        "semantic_results": reranked_results,
        "wisdom_results": wisdom_results,
    }


# ---------------------------------------------------------
# Prepare wisdom for Gemini prompt
# ---------------------------------------------------------

def prepare_wisdom_for_prompt(wisdom_results):
    wisdom = []

    for result in wisdom_results:

        record = result["record"]

        wisdom.append({
            "score": result["score"],
            "matched_terms": result["matched_terms"],
            "source": record.get("source", ""),
            "characters": record.get("characters", []),
            "situation": record.get("situation", ""),
            "lesson": record.get("lesson", ""),
            "interpretation": record.get("interpretation", ""),
            "caution": record.get("caution", ""),
        })

    return wisdom


# ---------------------------------------------------------
# Prepare passages for Gemini prompt
# ---------------------------------------------------------

def prepare_passages_for_prompt(semantic_results):
    passages = []

    for result in semantic_results:

        chunk = result["chunk"]

        passages.append({
            "score": result["score"],
            "wisdom_boost": result["wisdom_boost"],
            "final_score": result["final_score"],
            "parva": chunk.get("parva", ""),
            "section": chunk.get("section", ""),
            "text": chunk.get("text", ""),
        })

    return passages


# ---------------------------------------------------------
# Display dilemma
# ---------------------------------------------------------

def display_dilemma(dilemma):

    print("\n")
    print("=" * 70)
    print("UNDERSTOOD DILEMMA")
    print("=" * 70)

    print("\nOriginal:")
    print(dilemma.original_text)

    print("\nEmotions:")
    print(dilemma.emotions)

    print("\nSituation:")
    print(dilemma.situation)

    print("\nDesired actions:")
    print(dilemma.desired_actions)

    print("\nUnderlying conflicts:")
    print(dilemma.underlying_conflicts)

    print("\nValues:")
    print(dilemma.values)


# ---------------------------------------------------------
# Display retrieval results
# ---------------------------------------------------------

def display_results(results):

    display_dilemma(
        results["dilemma"]
    )

    # -----------------------------------------------------
    # Retrieval Query
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RETRIEVAL QUERY")
    print("=" * 70)

    print("\n")
    print(results["retrieval_query"])

    # -----------------------------------------------------
    # Semantic Results
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RELEVANT MAHABHARATA PASSAGES")
    print("=" * 70)

    for index, result in enumerate(
        results["semantic_results"],
        start=1,
    ):
        chunk = result["chunk"]

        print("\n" + "-" * 70)

        print(
            f"#{index} | "
            f"Semantic: {result['score']:.4f} | "
            f"Wisdom boost: {result['wisdom_boost']:.4f} | "
            f"Final: {result['final_score']:.4f}"
        )

        print(
            f"Source: "
            f"{chunk['parva']}, "
            f"Section {chunk['section']}"
        )

        print()

        print(chunk["text"])

    # -----------------------------------------------------
    # Wisdom Results
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RELEVANT WISDOM")
    print("=" * 70)

    for index, result in enumerate(
        results["wisdom_results"],
        start=1,
    ):
        record = result["record"]

        print("\n" + "-" * 70)

        print(
            f"#{index} | "
            f"Keyword score: {result['score']}"
        )

        print(
            f"Matched terms: "
            f"{', '.join(result['matched_terms'])}"
        )

        print(
            f"Source: "
            f"{record['source']}"
        )

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
# Display reasoning
# ---------------------------------------------------------

def display_reasoning(reasoning_result):

    print("\n")
    print("=" * 70)
    print("REASONING ENGINE")
    print("=" * 70)

    print("\nReasoning points:")

    for index, point in enumerate(
        reasoning_result.reasoning_points,
        start=1,
    ):
        print(
            f"{index}. {point}"
        )

    print("\nResponse guidance:")

    for index, guidance in enumerate(
        reasoning_result.response_guidance,
        start=1,
    ):
        print(
            f"{index}. {guidance}"
        )


# ---------------------------------------------------------
# Generate Dharma AI response
# ---------------------------------------------------------

def generate_dharma_response(
    reasoning_result,
):
    wisdom = prepare_wisdom_for_prompt(
        reasoning_result.selected_wisdom
    )

    passages = prepare_passages_for_prompt(
        reasoning_result.selected_passages
    )

    prompt = build_prompt(
        dilemma=reasoning_result.dilemma,
        wisdom=wisdom,
        passages=passages,
        reasoning_points=reasoning_result.reasoning_points,
    )

    gemini = GeminiClient()

    return gemini.generate(prompt)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Loading Mahabharata chunks...")

    chunks = load_chunks()

    print("Loading embeddings...")

    embeddings = load_embeddings()

    print("Loading Wisdom Map...")

    wisdom_records = load_wisdom()

    print("\n" + "=" * 70)
    print("DHARMA AI — DILEMMA RETRIEVAL ENGINE")
    print("=" * 70)

    print(
        f"\nMahabharata chunks: "
        f"{len(chunks)}"
    )

    print(
        f"Embeddings: "
        f"{embeddings.shape}"
    )

    print(
        f"Wisdom records: "
        f"{len(wisdom_records)}"
    )

    # -----------------------------------------------------
    # Interactive loop
    # -----------------------------------------------------

    while True:

        text = input(
            "\nEnter your dilemma "
            "(or type 'exit'): "
        ).strip()

        if text.lower() == "exit":
            print("Goodbye.")
            break

        if not text:
            continue

        # -------------------------------------------------
        # Step 1 — Understand dilemma
        # -------------------------------------------------

        dilemma = analyze_dilemma(
            text
        )

        # -------------------------------------------------
        # Step 2 — Retrieve material
        # -------------------------------------------------

        results = retrieve_dilemma(
            dilemma,
            model,
            chunks,
            embeddings,
            wisdom_records,
        )

        display_results(
            results
        )

        # -------------------------------------------------
        # Step 3 — Reason over material
        # -------------------------------------------------

        reasoning_result = reason(
            dilemma,
            results["wisdom_results"],
            results["semantic_results"],
        )

        display_reasoning(
            reasoning_result
        )

        # -------------------------------------------------
        # Step 4 — Generate Gemini response
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("DHARMA AI RESPONSE")
        print("=" * 70)

        try:
            response = generate_dharma_response(
                reasoning_result
            )

            print("\n")
            print(response)

        except Exception as error:
            print("\nGemini request failed:")
            print(error)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()