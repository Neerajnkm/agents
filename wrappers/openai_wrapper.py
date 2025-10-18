import os
from dotenv import load_dotenv
import openai
from typing import Any, Dict, List, Optional

load_dotenv()

class OpenAIWrapper:
    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.default_model = default_model or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo")
        openai.api_key = self.api_key

    def generate(self, prompt: str, images: Optional[List[Any]] = None, hyper_parameters: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> str:
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
