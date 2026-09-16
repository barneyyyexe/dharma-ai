from dataclasses import dataclass, asdict


@dataclass
class Dilemma:
    original_text: str
    emotions: list[str]
    situation: list[str]
    desired_actions: list[str]
    underlying_conflicts: list[str]
    values: list[str]


# ---------------------------------------------------------
# Keyword groups
# ---------------------------------------------------------

EMOTION_KEYWORDS = {
    "anger": [
        "angry",
        "anger",
        "furious",
        "rage",
        "mad",
        "wrath",
        "resentment",
    ],
    "hurt": [
        "hurt",
        "pain",
        "painful",
        "wounded",
    ],
    "fear": [
        "afraid",
        "fear",
        "scared",
        "worried",
        "anxious",
    ],
    "sadness": [
        "sad",
        "sadness",
        "heartbroken",
        "grief",
        "lonely",
    ],
    "guilt": [
        "guilty",
        "guilt",
        "regret",
        "ashamed",
    ],
    "confusion": [
        "confused",
        "confusion",
        "unsure",
        "uncertain",
        "don't know",
        "dont know",
    ],
}


SITUATION_KEYWORDS = {
    "betrayal": [
        "betrayed",
        "betrayal",
        "backstabbed",
        "backstab",
        "cheated on",
    ],
    "relationship_conflict": [
        "partner",
        "relationship",
        "girlfriend",
        "boyfriend",
        "husband",
        "wife",
    ],
    "family_conflict": [
        "family",
        "father",
        "mother",
        "brother",
        "sister",
        "parent",
    ],
    "work_conflict": [
        "boss",
        "manager",
        "colleague",
        "coworker",
        "office",
        "work",
        "job",
    ],
    "insult": [
        "insult",
        "insulted",
        "humiliated",
        "disrespected",
        "disrespect",
    ],
    "loss": [
        "lost",
        "loss",
        "died",
        "death",
    ],
}


ACTION_KEYWORDS = {
    "revenge": [
        "revenge",
        "retaliate",
        "retaliation",
        "get back",
        "pay back",
        "punish",
    ],
    "confrontation": [
        "confront",
        "confrontation",
        "fight",
        "argue",
        "attack",
    ],
    "forgiveness": [
        "forgive",
        "forgiveness",
        "let go",
        "move on",
    ],
    "withdrawal": [
        "leave",
        "walk away",
        "distance myself",
        "cut them off",
    ],
    "reconciliation": [
        "reconcile",
        "make peace",
        "fix things",
        "repair",
    ],
}


# ---------------------------------------------------------
# Conflict rules
# ---------------------------------------------------------

def detect_conflicts(emotions, situations, actions):
    conflicts = []

    if "anger" in emotions and "revenge" in actions:
        conflicts.append("anger_vs_self_control")

    if "betrayal" in situations and "revenge" in actions:
        conflicts.append("justice_vs_revenge")

    if "relationship_conflict" in situations:
        if "withdrawal" in actions:
            conflicts.append("relationship_vs_self_protection")

    if "family_conflict" in situations:
        conflicts.append("family_vs_personal_values")

    if "insult" in situations and "confrontation" in actions:
        conflicts.append("dignity_vs_restraint")

    if "forgiveness" in actions and "betrayal" in situations:
        conflicts.append("forgiveness_vs_self_protection")

    return conflicts


# ---------------------------------------------------------
# Value inference
# ---------------------------------------------------------

def infer_values(
    emotions,
    situations,
    actions,
    conflicts,
):
    values = []

    if "anger_vs_self_control" in conflicts:
        values.extend([
            "self_control",
            "dignity",
        ])

    if "justice_vs_revenge" in conflicts:
        values.extend([
            "justice",
            "self_control",
        ])

    if "relationship_vs_self_protection" in conflicts:
        values.extend([
            "relationship",
            "self_protection",
        ])

    if "family_vs_personal_values" in conflicts:
        values.extend([
            "family",
            "personal_values",
        ])

    if "dignity_vs_restraint" in conflicts:
        values.extend([
            "dignity",
            "restraint",
        ])

    if "forgiveness_vs_self_protection" in conflicts:
        values.extend([
            "forgiveness",
            "self_protection",
        ])

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(values))


# ---------------------------------------------------------
# Generic keyword detector
# ---------------------------------------------------------

def detect_categories(text, keyword_groups):
    text = text.lower()

    detected = []

    for category, keywords in keyword_groups.items():

        for keyword in keywords:

            if keyword in text:
                detected.append(category)
                break

    return detected


# ---------------------------------------------------------
# Analyze dilemma
# ---------------------------------------------------------

def analyze_dilemma(text):
    emotions = detect_categories(
        text,
        EMOTION_KEYWORDS,
    )

    situations = detect_categories(
        text,
        SITUATION_KEYWORDS,
    )

    actions = detect_categories(
        text,
        ACTION_KEYWORDS,
    )

    conflicts = detect_conflicts(
        emotions,
        situations,
        actions,
    )

    values = infer_values(
        emotions,
        situations,
        actions,
        conflicts,
    )

    return Dilemma(
        original_text=text,
        emotions=emotions,
        situation=situations,
        desired_actions=actions,
        underlying_conflicts=conflicts,
        values=values,
    )


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

def display_dilemma(dilemma):
    data = asdict(dilemma)

    print()
    print("=" * 70)
    print("DILEMMA ANALYSIS")
    print("=" * 70)

    print("\nOriginal:")
    print(data["original_text"])

    print("\nEmotions:")
    print(data["emotions"])

    print("\nSituation:")
    print(data["situation"])

    print("\nDesired actions:")
    print(data["desired_actions"])

    print("\nUnderlying conflicts:")
    print(data["underlying_conflicts"])

    print("\nValues:")
    print(data["values"])


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("DHARMA AI — DILEMMA ANALYZER")
    print("=" * 70)

    print("\nThis is an initial rule-based prototype.")
    print("Type 'exit' to quit.")

    while True:

        text = input(
            "\nEnter your dilemma: "
        ).strip()

        if text.lower() == "exit":
            print("Goodbye.")
            break

        if not text:
            continue

        dilemma = analyze_dilemma(text)

        display_dilemma(dilemma)


if __name__ == "__main__":
    main()