import os
from wrappers import LlamaWrapper, GroqWrapper, HuggingFaceWrapper


class LLMCentralCaller:
    """
    Centralized interface for calling different LLM providers (Llama, Groq, HuggingFace).

    Methods:
        call(prompt, **kwargs): Generate a response from the selected LLM provider.
    """
    def __init__(self, provider=None):
        """
        Initialize the LLMCentralCaller and select the LLM provider based on argument or environment variable.

        Args:
            provider (str, optional): The LLM provider to use ('llama', 'groq', or 'huggingface'). Defaults to environment variable LLM_PROVIDER or 'llama'.

        Raises:
            ValueError: If the provider is unknown.
        """
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
        """
        Generate a response from the selected LLM provider for the given prompt and additional arguments.

        Args:
            prompt (str): The prompt to send to the LLM.
            **kwargs: Additional keyword arguments passed to the provider's generate method.

        Returns:
            str: The generated response from the LLM provider.
        """
        return self.llm.generate(prompt, **kwargs)