import requests
from config.settings import settings

def generate(prompt: str, provider: str = "ollama", model: str | None = None) -> str:
    """Provider adapter. Add new vendors here without changing chat orchestration."""
    provider = (provider or "ollama").lower()
    if provider == "ollama":
        selected = model or settings.ollama_model
        try:
            r = requests.post(f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                json={"model": selected, "prompt": prompt, "stream": False}, timeout=180)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama unavailable at {settings.ollama_base_url}. Start Ollama and pull the model '{selected}'. Details: {exc}") from exc
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("Set OPENAI_API_KEY in .env before choosing OpenAI.")
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        result = client.chat.completions.create(model=model or settings.openai_model,
            messages=[{"role": "user", "content": prompt}])
        return result.choices[0].message.content or ""
    raise ValueError(f"Unknown provider: {provider}")
