from pathlib import Path
import json
import re


INPUT_FILE = Path("data/processed/adi_parva.jsonl")
OUTPUT_FILE = Path("data/processed/adi_parva_chunks.jsonl")

TARGET_CHARS = 2000
MAX_CHARS = 3000


def split_large_paragraph(text: str):
    """Split unusually large paragraphs at sentence boundaries."""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    pieces = []
    current = ""

    for sentence in sentences:

        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= MAX_CHARS:
            current = candidate
        else:
            if current:
                pieces.append(current)

            current = sentence

    if current:
        pieces.append(current)

    return pieces


def remove_section_ending(text: str) -> str:
    """Remove Gutenberg section-ending boilerplate from anywhere in a paragraph."""

    patterns = [
        r"\s*And so ends the .*?(?:Adi Parva|Mahabharata)\.?$",
        r"\s*And thus ends the .*?(?:Adi Parva|Mahabharata)\.?$",
        r"\s*So ends the .*?(?:Adi Parva|Mahabharata)\.?$",
        r"\s*Thus endeth the .*?(?:Adi Parva|Mahabharata)\.?$",
    ]

    cleaned = text

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    return cleaned.strip()


def create_chunks(section):
    """Create retrieval-friendly chunks while preserving provenance."""

    paragraphs = section["paragraphs"]

    # Remove section-ending boilerplate.
    cleaned_paragraphs = []

    for paragraph in paragraphs:

        paragraph = remove_section_ending(paragraph)

        if paragraph:
            cleaned_paragraphs.append(paragraph)

    paragraphs = cleaned_paragraphs

    # Split unusually large paragraphs at sentence boundaries.
    normalized = []

    for paragraph_number, paragraph in enumerate(paragraphs, 1):

        if len(paragraph) <= MAX_CHARS:

            normalized.append(
                (
                    paragraph_number,
                    paragraph,
                )
            )

        else:

            pieces = split_large_paragraph(paragraph)

            for piece_number, piece in enumerate(pieces, 1):

                normalized.append(
                    (
                        f"{paragraph_number}.{piece_number}",
                        piece,
                    )
                )

    chunks = []

    current_parts = []
    current_length = 0

    first_paragraph = None
    last_paragraph = None

    for paragraph_number, paragraph in normalized:

        paragraph_length = len(paragraph)

        # Finish the current chunk before exceeding MAX_CHARS.
        if (
            current_parts
            and current_length + paragraph_length + 2 > MAX_CHARS
        ):

            chunks.append(
                {
                    "paragraph_start": first_paragraph,
                    "paragraph_end": last_paragraph,
                    "text": "\n\n".join(current_parts),
                }
            )

            current_parts = []
            current_length = 0
            first_paragraph = None
            last_paragraph = None

        if first_paragraph is None:
            first_paragraph = paragraph_number

        current_parts.append(paragraph)

        current_length += paragraph_length + 2

        last_paragraph = paragraph_number

        # Finish the chunk once the target size is reached.
        if current_length >= TARGET_CHARS:

            chunks.append(
                {
                    "paragraph_start": first_paragraph,
                    "paragraph_end": last_paragraph,
                    "text": "\n\n".join(current_parts),
                }
            )

            current_parts = []
            current_length = 0
            first_paragraph = None
            last_paragraph = None

    # Add remaining content.
    if current_parts:

        chunks.append(
            {
                "paragraph_start": first_paragraph,
                "paragraph_end": last_paragraph,
                "text": "\n\n".join(current_parts),
            }
        )

    return chunks


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    sections = []

    with INPUT_FILE.open(
        encoding="utf-8"
    ) as file:

        for line in file:
            sections.append(
                json.loads(line)
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_chunks = 0

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:

        for section in sections:

            chunks = create_chunks(section)

            for index, chunk in enumerate(chunks, 1):

                record = {
                    "id": (
                        f"adi-{section['section_number']:03d}"
                        f"-chunk-{index:03d}"
                    ),
                    "parva": section["parva"],
                    "section": section["section"],
                    "section_number": section["section_number"],
                    "chunk_number": index,
                    "paragraph_start": chunk["paragraph_start"],
                    "paragraph_end": chunk["paragraph_end"],
                    "source": section["source"],
                    "text": chunk["text"],
                    "characters": len(chunk["text"]),
                }

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                total_chunks += 1

    print("Chunking complete.")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total chunks: {total_chunks}")


if __name__ == "__main__":
    main()