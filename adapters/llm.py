"""Wraps the LLM client so services never talk to a vendor SDK directly.

Defaults to Anthropic's Claude. Swapping providers means rewriting this one file.
"""

import os

import anthropic

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["LLM_API_KEY"])
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
