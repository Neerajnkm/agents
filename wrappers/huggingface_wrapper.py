import os
from dotenv import load_dotenv
import requests
from typing import Any, Dict, List, Optional

load_dotenv()

class HuggingFaceWrapper:
    """
    Wrapper for interacting with the HuggingFace Inference API for text (and optionally image) generation.

    Methods:
        generate(prompt, images=None, hyper_parameters=None, model=None): Generate a response from the HuggingFace model.
    """
    def __init__(self, api_key: Optional[str] = None, model_id: Optional[str] = None):
        """
        Initialize the HuggingFaceWrapper.

        Args:
            api_key (str, optional): HuggingFace API key. If not provided, loaded from environment.
            model_id (str, optional): Default model ID. If not provided, loaded from environment.
        """
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.model_id = model_id or os.getenv("HUGGINGFACE_DEFAULT_MODEL", "bigscience/bloom")
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"

    def generate(self, prompt: str, images: Optional[List[Any]] = None, hyper_parameters: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> str:
        """
        Generate a response from the HuggingFace model for the given prompt and optional images/hyperparameters.

        Args:
            prompt (str): The prompt to send to the model.
            images (list, optional): List of images to include (if supported).
            hyper_parameters (dict, optional): Additional generation parameters.
            model (str, optional): Model name to use.

        Returns:
            str: The generated response or error message.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": prompt}
        if model:
            payload["model"] = model
        if hyper_parameters:
            payload["parameters"] = hyper_parameters
        if images:
            # Add image support if model supports it
            pass
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            elif isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                return result[0]["generated_text"]
            return str(result)
        except requests.exceptions.RequestException as e:
            return f"HuggingFace API request error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
