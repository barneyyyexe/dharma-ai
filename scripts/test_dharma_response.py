from app.ai.gemini import GeminiClient
from app.ai.prompt_builder import build_prompt
from app.reasoning.engine import reason
from scripts.dilemma_analyzer import analyze_dilemma


def main():
    user_text = (
        "I am angry with someone who betrayed me. "
        "I want revenge, but I am not sure whether it is the right thing to do."
    )

    # 1. Understand the user's dilemma
    dilemma = analyze_dilemma(user_text)

    # 2. Temporary test data
    # We will connect the real retrieval pipeline next.
    wisdom = [
        {
            "source": "Adi Parva, Section 79",
            "characters": ["Sukra", "Devayani"],
            "situation": (
                "A person is provoked by harmful words or actions "
                "and must exercise control over anger."
            ),
            "lesson": (
                "Being wronged or provoked does not require an immediate "
                "reaction. Controlling anger can be a form of strength."
            ),
            "interpretation": (
                "Restraint can help separate the wrong that occurred "
                "from the decision about how to respond."
            ),
            "caution": (
                "Restraint should not be interpreted as an obligation "
                "to tolerate abuse, injustice, or repeated wrongdoing."
            ),
        }
    ]

    passages = [
        {
            "parva": "Adi Parva",
            "section": "LXXIX",
            "text": (
                "A relevant passage concerning anger, provocation, "
                "and self-control."
            ),
        }
    ]

    # 3. Generate structured reasoning
    reasoning_result = reason(
        dilemma=dilemma,
        wisdom_results=wisdom,
        source_results=passages,
    )

    # 4. Build the Gemini prompt
    prompt = build_prompt(
        dilemma=reasoning_result.dilemma,
        wisdom=reasoning_result.selected_wisdom,
        passages=reasoning_result.selected_passages,
        reasoning_points=reasoning_result.reasoning_points,
    )

    # 5. Ask Gemini to generate the final response
    gemini = GeminiClient()
    response = gemini.generate(prompt)

    print()
    print("=" * 70)
    print("DHARMA AI RESPONSE")
    print("=" * 70)
    print()
    print(response)
    print()


if __name__ == "__main__":
    main()