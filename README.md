# KK2-Oraklet-SmolLM: How to run the project

**Run the following code:**
`uv venv`
`uv init`
`uv pip install "fastapi[standard]"`
`uv add "torch==2.2.2" transformers pydantic "fastapi[standard]" pandas python-dotenv` (This is specifically for Intel-based computer that support only up to Torch 2.2.2 and not any version above.)
`uv add pytest --dev`

How to use in Swagger UI: 1. Open http://localhost:8000/docs 2. Click POST /data/upload → Try it out 3. Click Choose File and select your CSV 4. Click Execute

    Errors:
      400 — file is not a CSV, is empty, or cannot be parsed
