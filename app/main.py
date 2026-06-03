# Defines all four API endpoints

import logging
from fastapi import FastAPI, UploadFile, File, HTTPException

from app.data import load_csv, get_stats, get_shape, get_top_bottom, get_zeros, get_missing, get_outliers, get_dataset, get_correlations
from app.schemas import (
    MissingResponse, ShapeResponse, TopBottomResponse, UploadResponse, StatsResponse, AskRequest, AskResponse, PromptBuilderInput, ZeroResponse, ZerosResponse,
    ShapeResponse, TopBottomResponse, ZerosResponse, MissingResponse, OutliersResponse,
)
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
      detail="File is too big. Max 10 MB allowed."
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

# Returns the number of rows and columns in the uploaded dataset
@app.get("/data/shape", response_model=ShapeResponse)
def data_shape():
  logger.info("shape request received")
  result = get_shape()
  logger.info(f"Shape: {result['rows']} rows, {result['columns']} columns")

# Returns the top n countries with the hghest happiness score
@app.get("/data/top", response_model=TopBottomResponse)
def data_top(n: int = 10):
    logger.info(f"Top {n} request received")
    result = get_top_bottom(n=n, order="top")
    logger.info(f"Returning top {n} countries")
    return result

# Returns te bottom n countries with the lowest happiness score
@app.get("/data/bottom", response_model=TopBottomResponse)
def data_bottom(n: int = 5):
    logger.info(f"Bottom {n} request received")
    result = get_top_bottom(n=n, order="bottom")
    logger.info(f"Returning bottom {n} countries")
    return result

# Finds countries with a value 0.000
@app.get("/data/zeros", response_model=ZerosResponse)
def data_zeros():
    logger.info("Zeros request received")
    result = get_zeros()
    logger.info(f"Zeros found in {len(result['zeros'])} columns")
    return result

# Return the count of NaN (null) values per column
@app.get("/data/missing", response_class=MissingResponse)
def data_missing():
   logger.info("Missing values request received")
   result = get_missing()
   logger.info(f"Missing values found in {len(result['missing'])} columns")

# Shows countries whose happiness score is higher or lower than GDP alone would predict
@app.get("/data/outliers", response_model=OutliersResponse)
def data_outliers(n: int = 5):
   logger.info(f"Outliers request received (n={n})")
   result = get_outliers(n=n)
   logger.info(
      f"Over-perfomers: {[c['country'] for c in result['over-performers']]} | "
      f"Under-performers: {[c['country'] for c in result['under-performers']]}"
   )
   return result


@app.get("/data/correlations")
def data_correlations():
  logger.info("Correlations request received")

  return get_correlations()

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

  # Run the full chain: PromptBuilder -> LLMRunner -> ResponseParser
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

