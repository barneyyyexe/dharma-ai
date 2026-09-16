from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ReasoningResult:
    dilemma: dict[str, Any]
    selected_wisdom: list[dict[str, Any]]
    selected_passages: list[dict[str, Any]]
    reasoning_points: list[str]
    response_guidance: list[str]


def select_wisdom(wisdom_results: list[dict[str, Any]], limit: int = 2):
    """
    Select the most relevant wisdom records.

    For the initial prototype, we simply use the
    keyword relevance score produced by the Wisdom Layer.
    """

    return sorted(
        wisdom_results,
        key=lambda item: item.get("score", 0),
        reverse=True
    )[:limit]


def select_passages(source_results: list[dict[str, Any]], limit: int = 3):
    """
    Select the highest-ranked source passages.

    This is intentionally simple for the first prototype.
    Later, a reasoning model will evaluate whether a passage
    is actually useful, safe, and contextually appropriate.
    """

    return sorted(
        source_results,
        key=lambda item: item.get("similarity", 0),
        reverse=True
    )[:limit]


def build_reasoning_points(dilemma, wisdom_results, source_results):
    """
    Build explicit reasoning points from the structured dilemma.

    This is not the final AI answer.
    It is an intermediate representation that a future
    reasoning model can use.
    """

    points = []

    if "anger_vs_self_control" in dilemma.underlying_conflicts:
        points.append(
            "Separate the feeling of anger from the decision about how to respond."
        )

    if "justice_vs_revenge" in dilemma.underlying_conflicts:
        points.append(
            "Consider whether the desired response seeks justice or is primarily driven by revenge."
        )

    if "betrayal" in dilemma.situation:
        points.append(
            "Acknowledge the harm caused by betrayal without assuming that retaliation is the only response."
        )

    if "self_control" in dilemma.values:
        points.append(
            "Evaluate whether acting immediately would allow anger to determine the response."
        )

    if "dignity" in dilemma.values:
        points.append(
            "Consider what response preserves personal dignity without unnecessary escalation."
        )

    if not points:
        points.append(
            "Consider the situation, competing values, possible consequences, and available responses."
        )

    return points


def build_response_guidance(dilemma):
    """
    Define how the future response should be framed.

    This is guidance for the response generator,
    not the response itself.
    """

    guidance = [
        "Acknowledge the person's situation and emotions.",
        "Do not dismiss or shame the person's feelings.",
        "Do not automatically endorse the desired action.",
        "Distinguish source material from modern interpretation.",
        "Use the retrieved wisdom as a lens rather than as an absolute command.",
        "Consider consequences and alternative responses.",
        "Leave the final decision with the user.",
    ]

    if "anger_vs_self_control" in dilemma.underlying_conflicts:
        guidance.append(
            "Encourage reflection before taking an action driven by anger."
        )

    if "justice_vs_revenge" in dilemma.underlying_conflicts:
        guidance.append(
            "Explore the difference between seeking legitimate resolution and seeking revenge."
        )

    return guidance


def reason(dilemma, wisdom_results, source_results):
    """
    Main reasoning pipeline.
    """

    selected_wisdom = select_wisdom(wisdom_results)
    selected_passages = select_passages(source_results)

    reasoning_points = build_reasoning_points(
        dilemma,
        selected_wisdom,
        selected_passages,
    )

    response_guidance = build_response_guidance(dilemma)

    return ReasoningResult(
        dilemma=asdict(dilemma),
        selected_wisdom=selected_wisdom,
        selected_passages=selected_passages,
        reasoning_points=reasoning_points,
        response_guidance=response_guidance,
    )
