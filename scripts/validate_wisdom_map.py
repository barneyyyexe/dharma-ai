from pathlib import Path
import json

INPUT_FILE = Path("data/knowledge/wisdom_map.jsonl")

REQUIRED_FIELDS = [
    "id",
    "source",
    "characters",
    "situation",
    "dilemma_types",
    "values_in_conflict",
    "emotions",
    "lesson",
    "interpretation",
    "caution",
    "relevance_to_modern_life",
    "keywords",
]


def validate_record(record, line_number):
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"missing field '{field}'")

    list_fields = [
        "characters",
        "dilemma_types",
        "values_in_conflict",
        "emotions",
        "relevance_to_modern_life",
        "keywords",
    ]

    for field in list_fields:
        if field in record and not isinstance(record[field], list):
            errors.append(f"'{field}' must be a list")

    text_fields = [
        "id",
        "source",
        "situation",
        "lesson",
        "interpretation",
        "caution",
    ]

    for field in text_fields:
        if field in record and not isinstance(record[field], str):
            errors.append(f"'{field}' must be a string")

    if errors:
        print(f"❌ Line {line_number}:")
        for error in errors:
            print(f"   - {error}")
        return False

    return True


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    valid_count = 0
    invalid_count = 0

    with INPUT_FILE.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"❌ Line {line_number}: Invalid JSON")
                print(f"   {error}")
                invalid_count += 1
                continue

            if validate_record(record, line_number):
                valid_count += 1
            else:
                invalid_count += 1

    print()
    print("=" * 60)
    print("WISDOM MAP VALIDATION")
    print("=" * 60)
    print(f"Valid records:   {valid_count}")
    print(f"Invalid records: {invalid_count}")

    if invalid_count == 0:
        print("\n✅ Wisdom map is valid.")
    else:
        print("\n❌ Wisdom map contains errors.")


if __name__ == "__main__":
    main()