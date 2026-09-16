from typing import Any


SYSTEM_INSTRUCTIONS = """
You are Dharma AI, a thoughtful conversational clarity engine.

Your purpose is to help a person think more clearly about a difficult situation.

You may use wisdom, experiences, conflicts, values, and consequences found
in the provided source material as a lens for reflection.

IMPORTANT PRINCIPLES:

1. Do not act like a guru, preacher, therapist, or fortune teller.
2. Do not tell the user what they must do.
3. Help the user understand the conflict and possible consequences.
4. Acknowledge the person's emotions without automatically endorsing their actions.
5. Do not assume that one response is universally correct.
6. Distinguish the source material from modern interpretation.
7. Do not force a connection to the source when the connection is weak.
8. Do not repeatedly mention the Mahabharata unless it is useful or the user asks.
9. Treat the retrieved wisdom as a lens, not an absolute command.
10. Preserve the user's agency. The final decision belongs to the user.
11. Do not use ancient teachings to justify abuse, violence, discrimination,
    manipulation, or harmful behavior.
12. When relevant, explore the difference between what a person feels,
    what they want to do, what they value, and what consequences may follow.

Write naturally and conversationally.

Do not mention internal systems such as:
- retrieval
- embeddings
- vector databases
- wisdom maps
- dilemma analyzers
- reasoning engines
- prompts

Do not invent source passages or teachings that are not provided.
"""


def build_prompt(
    dilemma: dict[str, Any],
    wisdom: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    reasoning_points: list[str],
) -> str:
    """
    Build the prompt sent to Gemini using the structured Dharma AI context.
    """

    emotions = ", ".join(dilemma.get("emotions", [])) or "Not clearly identified"
    situations = ", ".join(dilemma.get("situation", [])) or "Not clearly identified"
    actions = ", ".join(dilemma.get("desired_actions", [])) or "Not clearly identified"
    conflicts = ", ".join(
        dilemma.get("underlying_conflicts", [])
    ) or "Not clearly identified"
    values = ", ".join(dilemma.get("values", [])) or "Not clearly identified"

    prompt_parts = [
        SYSTEM_INSTRUCTIONS.strip(),
        "",
        "USER'S ORIGINAL SITUATION",
        "-------------------------",
        dilemma.get("original_text", ""),
        "",
        "UNDERSTOOD DILEMMA",
        "-----------------",
        f"Emotions: {emotions}",
        f"Situation: {situations}",
        f"Desired actions: {actions}",
        f"Underlying conflicts: {conflicts}",
        f"Values involved: {values}",
        "",
        "RELEVANT WISDOM",
        "---------------",
    ]

    if wisdom:
        for index, item in enumerate(wisdom, start=1):
            prompt_parts.extend(
                [
                    f"",
                    f"Wisdom {index}:",
                    f"Source: {item.get('source', 'Unknown')}",
                    f"Characters: {', '.join(item.get('characters', []))}",
                    f"Situation: {item.get('situation', '')}",
                    f"Lesson: {item.get('lesson', '')}",
                    f"Interpretation: {item.get('interpretation', '')}",
                    f"Caution: {item.get('caution', '')}",
                ]
            )
    else:
        prompt_parts.append("No structured wisdom was found.")

    prompt_parts.extend(
        [
            "",
            "RELEVANT SOURCE PASSAGES",
            "------------------------",
        ]
    )

    if passages:
        for index, passage in enumerate(passages, start=1):
            prompt_parts.extend(
                [
                    "",
                    f"Passage {index}:",
                    f"Source: {passage.get('parva', '')}, "
                    f"Section {passage.get('section', '')}",
                    f"Text: {passage.get('text', '')}",
                ]
            )
    else:
        prompt_parts.append("No source passages were found.")

    prompt_parts.extend(
        [
            "",
            "REASONING POINTS",
            "----------------",
        ]
    )

    if reasoning_points:
        for point in reasoning_points:
            prompt_parts.append(f"- {point}")
    else:
        prompt_parts.append("- Consider the situation, values, and consequences carefully.")

    prompt_parts.extend(
        [
            "",
            "TASK",
            "----",
            "Respond to the user naturally and thoughtfully.",
            "",
            "Your response should generally:",
            "- acknowledge what the person is experiencing",
            "- identify the central conflict when useful",
            "- use the provided wisdom as a lens",
            "- distinguish reflection from certainty",
            "- consider consequences and alternative responses",
            "- leave the decision with the user",
            "",
            "Do not simply summarize the source material.",
            "Do not give a generic motivational speech.",
            "Answer the person's actual situation.",
        ]
    )

    return "\n".join(prompt_parts)