import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


class GeminiClient:
    """Client for generating Dharma AI responses with Gemini."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3.6-flash"

    def generate(self, prompt: str) -> str:
        """Generate a response from Gemini."""

        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
        )

        return interaction.output_text.strip()