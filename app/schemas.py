from pydantic import BaseModel

# Returned by POST /data/upload
# Gives basic metadata about the dataset from uploaded CV file
class UploadResponse(BaseModel):
  rows: int
  columns: list[str]
  dtypes: dict[str, str]

# Returned by GET /data/stats
# Contains output of pandas describe() as a nested dictionary
class Statsresponse(BaseModel):
  stats: dict[str, dict[str, float]]

# Received by POST /ai/ask
# Sends user's question about the uploaded dataset
class AskRequest(BaseModel):
  question: str

# Returned by POST /ai/ask
# Contains question, the model's answer, and which was used
class AskResponse(BaseModel):
  question: str
  answer: str
  model: str

# Input to the PromptBuilder step
# Contains the user's question and the dataset statistics
class PromptBuilderInput(BaseModel):
  question: str
  stats: dict[str, dict[str, float]]

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