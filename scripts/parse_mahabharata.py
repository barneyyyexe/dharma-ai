from pathlib import Path
import json
import re


INPUT_FILE = Path("data/raw/ganguli/adi_parva.txt")
OUTPUT_FILE = Path("data/processed/adi_parva.jsonl")


def roman_to_int(roman: str) -> int:
    values = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }

    total = 0
    previous = 0

    for char in reversed(roman):
        value = values[char]

        if value < previous:
            total -= value
        else:
            total += value

        previous = value

    return total


def clean_paragraph(text: str) -> str:
    """Clean Gutenberg line wrapping while preserving paragraphs."""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_sections(text: str):
    """Extract sections while preserving paragraph boundaries."""

    section_pattern = re.compile(
        r"^SECTION ([IVXLCDM]+)\s*$",
        re.MULTILINE,
    )

    matches = list(section_pattern.finditer(text))

    records = []

    for index, match in enumerate(matches):

        roman_section = match.group(1)
        section_number = roman_to_int(roman_section)

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        section_raw = text[start:end].strip()

        raw_paragraphs = re.split(
            r"\n\s*\n",
            section_raw,
        )

        paragraphs = []

        for paragraph in raw_paragraphs:

            paragraph = clean_paragraph(paragraph)

            if paragraph:
                paragraphs.append(paragraph)

        record = {
            "id": f"adi-{section_number:03d}",
            "parva": "Adi Parva",
            "section": roman_section,
            "section_number": section_number,
            "source": "Kisari Mohan Ganguli",
            "paragraphs": paragraphs,
            "paragraph_count": len(paragraphs),
        }

        records.append(record)

    return records


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_text = INPUT_FILE.read_text(
        encoding="utf-8"
    )

    records = parse_sections(raw_text)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("Parsing complete.")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Sections found: {len(records)}")


if __name__ == "__main__":
    main()