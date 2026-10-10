
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Project root: parent of the config directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load the project's .env file explicitly
load_dotenv(dotenv_path=ENV_FILE, override=True)


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "AI Learning Lab")
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")

    default_llm_provider: str = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    chroma_path: str = os.getenv("CHROMA_PATH", "./data/chroma")
    pdf_directory: str = os.getenv("PDF_DIRECTORY", "./data/pdfs")

    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_database: str = os.getenv(
        "MYSQL_DATABASE", "first_application_ai"
    )


settings = Settings()

# Safe diagnostic output: do not print passwords or API keys
print(f".env file exists: {ENV_FILE.is_file()}")
print(f"MYSQL_HOST: {settings.mysql_host}")
print(f"MYSQL_PORT: {settings.mysql_port}")
print(f"MYSQL_USER: {settings.mysql_user}")
print(f"MYSQL_DATABASE: {settings.mysql_database}")
