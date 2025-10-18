import os
from dotenv import load_dotenv
import requests
from typing import Any, Dict, List, Optional

load_dotenv()

class GroqWrapper:
    """
    Wrapper for interacting with the Groq API for text (and optionally image) generation.

    Methods:
        generate(prompt, images=None, hyper_parameters=None, model=None): Generate a response from the Groq model.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the GroqWrapper.

        Args:
            api_key (str, optional): Groq API key. If not provided, loaded from environment.
            model (str, optional): Default model name. If not provided, loaded from environment.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_DEFAULT_MODEL", "llama-2-70b-chat")
        self.api_url = f"https://api.groq.com/openai/v1/chat/completions"

    def generate(self, prompt: str, images: Optional[List[Any]] = None, hyper_parameters: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> str:
        """
        Generate a response from the Groq model for the given prompt and optional images/hyperparameters.

        Args:
            prompt (str): The prompt to send to the model.
            images (list, optional): List of images to include (if supported).
            hyper_parameters (dict, optional): Additional generation parameters.
            model (str, optional): Model name to use.

        Returns:
            str: The generated response or error message.
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": hyper_parameters.get("temperature", 0.7) if hyper_parameters else 0.7,
            "top_p": hyper_parameters.get("top_p", 1.0) if hyper_parameters else 1.0,
        }
        if images:
            # Add image support if model supports it
            pass
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            try:
                return result["choices"][0]["message"]["content"].strip()
            except Exception:
                return str(result)
        except requests.exceptions.RequestException as e:
            return f"Groq API request error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
