from pydantic import BaseModel, Field

# Returned by POST /data/upload
# Gives basic metadata about the dataset from uploaded CV file
class UploadResponse(BaseModel):
  rows: int
  columns: list[str]
  dtypes: dict[str, str]

# Returned by GET /data/stats
# Contains output of pandas describe() as a nested dictionary
class StatsResponse(BaseModel):
  stats: dict[str, dict[str, float]]

# Received by POST /ai/ask
# Sends user's question about the uploaded dataset, cannot be empty string
class AskRequest(BaseModel):
  question: str = Field(min_length=5, max_length=500)

# Returned by POST /ai/ask
# Contains question, the model's answer, and which was used
class AskResponse(BaseModel):
  question: str
  answer: str
  model: str

# Returned by GET /data/shape
# contains the number of rows and columns in the dataset (df.shape)
class ShapeResponse(BaseModel):
  rows: int
  columns: int

# Gives a single country and its happiness score
class CountryScore(BaseModel):
  country: str
  happiness_score: float

# Returned by GET /data/top and GET /data/bottom
# contains a ranked (ordered) list of countries with their happiness scores
class TopBottomResponse(BaseModel):
  results: list[CountryScore]

# Returned by GET /data/zeros, gives country that have 0.000 values
class ZerosResponse(BaseModel):
  zeros: dict[str, list[str]]

# Returned by GET /data/missing
# Gives how many NaN values each column has (mirrors df.isnull())
class MissingResponse(BaseModel):
  missing: dict[str, int]

# Shows how each variable correlates with happiness score
# Returned by GET /data/correlations
class CorrelationsResponse(BaseModel):
  correlations: dict[str, float]

# Returned by GET /data/outliers
# Shows countries whose happiness score differs most from what their GDP alone would predict.
class OutliersResponse(BaseModel):
  over_performers: list[CountryScore]
  under_performers: list[CountryScore]    # less happy than GDP predicts


# Input to the PromptBuilder step
# Contains the user's question and the dataset statistics
class PromptBuilderInput(BaseModel):
  question: str
  stats: dict[str, dict[str, float]]
  top: list[dict]
  bottom: list[dict]
  correlations: dict[str, float]
  outliers: dict[str, list[dict]]

# Output of PromptBuilder, and input to LLMRunner
# Contains formatted prompt ready to be sent to SmolLLM and carries forwarded questions
class PromptBuilderOutput(BaseModel):
  prompt: str
  question: str

# Output of LLMRunner, and input to ResponseParser
# Contains raw text that SmolLLM generated and carries forwarded question
class LLMRunnerOutput(BaseModel):
  raw_output: str
  question: str

# Output of ResponseParser
# Maps directly onto AskResponse and gives carries what is returned to user
class ResponseParserOutput(BaseModel):
  question: str
  answer: str
  model: str