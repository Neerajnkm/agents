import os
from dotenv import load_dotenv
import requests
from typing import Any, Dict, List, Optional
import json

load_dotenv()

class LlamaWrapper:
    def __init__(self, api_url: Optional[str] = None, default_model: Optional[str] = None):
        self.api_url = api_url or os.getenv("LLAMA_API_URL")
        self.default_model = default_model or os.getenv("LLAMA_DEFAULT_MODEL", "llama2")

    def generate(self, prompt: str, images: Optional[List[Any]] = None, hyper_parameters: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> str:
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
