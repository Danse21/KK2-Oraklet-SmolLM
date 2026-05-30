# Defines all four API endpoints

import logging
from fastapi import FastAPI, UploadFile, File, HTTPException

from app.data import load_csv, get_stats, get_dataset
from app.schemas import UploadResponse, StatsResponse, AskRequest, AskResponse, PromptBuilderInput
from app.chain.pipeline import oraklet

# Record (log) what happens, INFO = normal operations (requests, responses)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
  title="The Oracle",
  description="A typed LLM chain with FastAPI och SolLLM",
  version="1.0.0",
)

# Returns status message to confirm server is running
@app.get("/health")
def health():
  logger.info("Health check called")
  return {"Status": "ok"}

# Uploads a CSV file
@app.post("/data/upload", response_model=UploadResponse)
async def upload_data(file: UploadFile = File(...)):
  logger.info(f"{file.filename}, successfully uploaded")

  # Read the uploaded file size
  file_size = await file.read()

  # Check that file size is not more than 10 MB
  max_size = 10 * 1024 * 1024
  if len(file_size) > max_size:
    raise HTTPException(
      status_code=400,
      detail="File is too big. Max 10 mB allowed."
    )
  
  # load_CSV validates, parses, renames columns and stores the dataset
  metadata = load_csv(file_size, file.filename)

  logger.info(f"Dataset conatins: {metadata['rows']} rows, {len(metadata['columns'])} columns")
  return metadata

# Returns descriptive statistics (mean, std, min, max, percentiles) 
@app.get("/data/stats", response_model=StatsResponse)
def data_stats():
  logger.info("Stats request received")

  # get_stats() calls get_dataset internally
  # where dataset is not found, raises HTTPException
  stats = get_stats()

  logger.info("Stats returned successfully")
  return stats

# Takes a natural language question about the uploaded dataset,
# runs it through the Runnable chain, and returns an AI-generated answer
@app.post("/ai/ask", response_model=AskResponse)
def ask(body: AskRequest):
  logger.info(f"AI ask request: '{body.question}'")

  # Get the current stats
  stats_data = get_stats()
  # Build the input for the first chain step (PromptBuilder)
  prompt_builder_input = PromptBuilderInput(
    question=body.question,
    stats=stats_data["stats"],
  )

  # Runn the full chain: PromptBuilder -> LLMRunner -> ResponseParser
  try:
    result = oraklet.invoke(prompt_builder_input)
    logger.info(f"Chain successfully completed. Answer: '{result.answer[:60]}...")
  except Exception as e:
    logger.error(f"Chain failed: {e}")
    raise HTTPException(
      status_code=500,
      detail=f"The model could not generate an answer: {e}"
    )
  return AskResponse(
    question=result.question,
    answer=result.answer,
    model=result.model,
  )

