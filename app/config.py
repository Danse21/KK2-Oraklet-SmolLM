# Loads settings from the .env file and exposes them to the rest of the app

import os
from dotenv import load_dotenv

# Load the .env file into the process environment
load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")  # GROQ API key

MODEL_NAME: str = os.getenv(
  "MODEL_NAME",
  "HuggingFaceTB/SmolLM2-135M-Instruct"
)
