import os
from dotenv import load_dotenv
import openai
from typing import Any, Dict, List, Optional

load_dotenv()

class OpenAIWrapper:
    """
    Wrapper for interacting with the OpenAI API for text (and optionally image) generation.

    Methods:
        generate(prompt, images=None, hyper_parameters=None, model=None): Generate a response from the OpenAI model.
    """
    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        """
        Initialize the OpenAIWrapper.

        Args:
            api_key (str, optional): OpenAI API key. If not provided, loaded from environment.
            default_model (str, optional): Default model name. If not provided, loaded from environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.default_model = default_model or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo")
        openai.api_key = self.api_key

    def generate(self, prompt: str, images: Optional[List[Any]] = None, hyper_parameters: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> str:
        """
        Generate a response from the OpenAI model for the given prompt and optional images/hyperparameters.

        Args:
            prompt (str): The prompt to send to the model.
            images (list, optional): List of images to include (if supported).
            hyper_parameters (dict, optional): Additional generation parameters (e.g., temperature, top_p).
            model (str, optional): Model name to use.

        Returns:
            str: The generated response or error message.
        """
        params = {
            "model": model or (hyper_parameters.get("model") if hyper_parameters else self.default_model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": hyper_parameters.get("temperature", 0.7) if hyper_parameters else 0.7,
            "top_p": hyper_parameters.get("top_p", 1.0) if hyper_parameters else 1.0,
        }
        if images:
            # Add image support if model supports it
            pass
        try:
            response = openai.ChatCompletion.create(**params)
            return response.choices[0].message['content'].strip()
        except openai.error.OpenAIError as e:
            return f"OpenAI API error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
