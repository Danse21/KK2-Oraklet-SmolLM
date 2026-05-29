# Loads settings from the .env file and exposes them to the rest of the app

import os
from dotenv import load_dotenv

# Load the .env file into the process environment
load_dotenv()

HF_TOKEN: str | None = os.getenv("HF_TOKEN")  # HuggingFace API token

MODEL_NAME: str = os.getenv(
  "MODEL_NAME",
  "HuggingFaceTB/SmolLM2-135M-Instruct"
)
