import os
from dotenv import load_dotenv
import requests
from typing import Any, Dict, List, Optional
import json

load_dotenv()

class LlamaWrapper:
    """
    Wrapper for interacting with a Llama API endpoint for text generation.

    Methods:
        generate(prompt, images=None, hyper_parameters=None, model=None): Generate a response from the Llama model.
    """
    def __init__(self, api_url: Optional[str] = None, default_model: Optional[str] = None):
        """
        Initialize the LlamaWrapper.

        Args:
            api_url (str, optional): Llama API endpoint URL. If not provided, loaded from environment.
            default_model (str, optional): Default model name. If not provided, loaded from environment.
        """
        self.api_url = api_url or os.getenv("LLAMA_API_URL")
        self.default_model = default_model or os.getenv("LLAMA_DEFAULT_MODEL", "llama2")

    def generate(self, prompt: str, images: Optional[List[Any]] = None, hyper_parameters: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> str:
        """
        Generate a response from the Llama model for the given prompt and optional images/hyperparameters.

        Args:
            prompt (str): The prompt to send to the model.
            images (list, optional): List of images to include (if supported).
            hyper_parameters (dict, optional): Additional generation parameters.
            model (str, optional): Model name to use.

        Returns:
            str: The generated response or error message.
        """
        data = {
            "model": model or self.default_model,
            "prompt": prompt,
        }
        if hyper_parameters and "temperature" in hyper_parameters:
            data["options"] = {"temperature": hyper_parameters["temperature"]}
        try:
            response = requests.post(self.api_url, json=data, stream=True)
            response.raise_for_status()
            answer = ""
            for line in response.iter_lines():
                if line:
                    try:
                        obj = json.loads(line.decode("utf-8"))
                        if "response" in obj:
                            answer += obj["response"]
                    except Exception:
                        continue
            return answer.strip() if answer.strip() else "No response from model."
        except requests.exceptions.RequestException as e:
            return f"Llama API request error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
