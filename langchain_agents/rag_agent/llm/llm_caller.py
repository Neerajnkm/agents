import os
from wrappers import LlamaWrapper, GroqWrapper, HuggingFaceWrapper


class LLMCentralCaller:
    def __init__(self, provider=None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "llama")
        if self.provider == "llama":
            self.llm = LlamaWrapper()
        elif self.provider == "groq":
            self.llm = GroqWrapper()
        elif self.provider == "huggingface":
            self.llm = HuggingFaceWrapper()
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def call(self, prompt, **kwargs):
        return self.llm.generate(prompt, **kwargs)