import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8501,http://127.0.0.1:3000,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]

# Authenticate with HuggingFace if token is provided
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# Model config
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Validate required keys
def validate_config():
    """Validate that all required environment variables are set."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Copy .env.example to .env and fill in your API keys."
        )
