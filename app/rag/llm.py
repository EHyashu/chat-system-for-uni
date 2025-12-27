from abc import ABC, abstractmethod
from typing import List

from app.config import Settings


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class DummyLLM(BaseLLM):
    """Fallback LLM for development without external API calls."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "I could not find this information in the university documents."


class OpenAILLM(BaseLLM):
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise ValueError("UNI_RAG_OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model_name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or "I could not find this information in the university documents."


def get_llm(settings: Settings) -> BaseLLM:
    if settings.llm_provider == "dummy":
        return DummyLLM()
    if settings.llm_provider == "openai":
        return OpenAILLM(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
