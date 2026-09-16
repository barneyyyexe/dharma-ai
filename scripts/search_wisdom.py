from pathlib import Path
import json

INPUT_FILE = Path("data/knowledge/wisdom_map.jsonl")


def load_wisdom_map():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    records = []

    with INPUT_FILE.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                )

            records.append(record)

    return records


def search_wisdom(query, records):
    query_words = set(query.lower().split())

    results = []

    for record in records:
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
            1 for word in query_words
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

    return results


def main():
    records = load_wisdom_map()

    print("=" * 70)
    print("WISDOM MAP SEARCH")
    print("=" * 70)
    print(f"Records loaded: {len(records)}")

    while True:
        query = input(
            "\nEnter a dilemma (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            print("Goodbye.")
            break

        if not query:
            continue

        results = search_wisdom(query, records)

        print("\nMatching wisdom:\n")

        if not results:
            print("No matching wisdom found.")
            continue

        for rank, result in enumerate(results, 1):
            record = result["record"]

            print("=" * 70)
            print(f"#{rank} | Score: {result['score']}")
            print(f"Source: {record['source']}")
            print(f"Characters: {', '.join(record['characters'])}")
            print(f"Dilemma: {', '.join(record['dilemma_types'])}")
            print()
            print("Situation:")
            print(record["situation"])
            print()
            print("Lesson:")
            print(record["lesson"])
            print()
            print("Interpretation:")
            print(record["interpretation"])
            print()
            print("Caution:")
            print(record["caution"])


if __name__ == "__main__":
    main()