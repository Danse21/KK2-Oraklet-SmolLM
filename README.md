# KK2-Oraklet-SmolLM: How to run the project

**Run the following code:**
`uv venv`
`uv init`
`uv pip install "fastapi[standard]"`
`uv add "torch==2.2.2" transformers pydantic "fastapi[standard]" pandas python-dotenv` (This is specifically for Intel-based computer that support only up to Torch 2.2.2 and not any version above.)
