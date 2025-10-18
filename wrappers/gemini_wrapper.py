import os
from dotenv import load_dotenv
import requests
from typing import Any, Dict, List, Optional

load_dotenv()

class GeminiWrapper:
    """
    Wrapper for interacting with the Google Gemini API for text (and optionally image) generation.

    Methods:
        generate(prompt, images=None, hyper_parameters=None, model=None): Generate a response from the Gemini model.
    """
    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        """
        Initialize the GeminiWrapper.

        Args:
            api_key (str, optional): Gemini API key. If not provided, loaded from environment.
            default_model (str, optional): Default model name. If not provided, loaded from environment.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.default_model = default_model or os.getenv("GEMINI_DEFAULT_MODEL", "gemini-pro")
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    def generate(self, prompt: str, images: Optional[List[Any]] = None, hyper_parameters: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> str:
        """
        Generate a response from the Gemini model for the given prompt and optional images/hyperparameters.

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
            "contents": [{"parts": [{"text": prompt}]}]
        }
        data["model"] = model or self.default_model
        if images:
            # Add image support if model supports it
            pass
        if hyper_parameters:
            data.update(hyper_parameters)
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            try:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                return str(result)
        except requests.exceptions.RequestException as e:
            return f"Gemini API request error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
