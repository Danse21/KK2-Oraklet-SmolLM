# the 3 chain steps: 
# PromptBuilder: (receives the user's question and the dataset statistics and formats them into a structured prompt),
# LLMRunner: (takes that formatted prompt and sends it to the SmolLLM model and generates text response ie., a raw and unprocessed output model)
# ResponseParser: (takes the raw model output and cleans it up and returns a clean, structured response that maps directly onto what the API sends back to the user.)

from typing import Any
from pydantic import PrivateAttr


from app.chain.runnable import Runnable
from app.schemas import (
  PromptBuilderInput,
  PromptBuilderOutput,
  LLMRunnerOutput,
  ResponseParserOutput,
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Step 1 in the chain:
# Takes user's question and dataset statistics (Input) and 
# formats them into a structured prompt (Output)
class PromptBuilder(Runnable[PromptBuilderInput, PromptBuilderOutput]):
  name: str = "prompt_builder"

  # Format the stats dictionary into readable text
  def invoke(self, data: PromptBuilderInput) -> PromptBuilderOutput:
    stats_lines = []
    for column, values in data.stats.items():
      formatted = ", ".join(
        f"{k}: {round(v, 2)}" for k, v in values.items()
      )
      stats_lines.append(f" {column}: {formatted}")
    stats_text = "\n".join(stats_lines)

    # For top and bottom happiness score questions
    top_lines = [
      f"{i+1}, {entry['country']}: {entry['happiness_score']}"
      for i, entry in enumerate(data.top)
    ]
    top_text = "\n".join(top_lines)

    bottom_lines = [
      f"{i+1}, {entry['country']}: {entry['happiness_score']}"
      for i, entry in enumerate(data.bottom)
    ]
    bottom_text = "\n".join(bottom_lines)

    # Sort correlation values from strongest to weakest
    sorted_corr = sorted(
      data.correlations.items(),
      key=lambda x: abs(x[1]),
      reverse=True,
    )
    corr_lines = [
      f"{col}: {round(corr, 3)}"
      for col, corr in sorted_corr
    ]
    corr_text = "\n".join(corr_lines)

    # Countries whose happiness score differs most from what their GDP would predict
    over_lines =[
      f"{entry['country']}: {entry['happiness_score']}"
      for entry in data.outliers.get("over_performers", [])
    ]
    under_lines = [
      f"{entry['country']}: {entry['happiness_score']}"
      for entry in data.outliers.get("under_performers", [])
    ]
    over_text = "\n".join(over_lines) or "None"
    under_text = "\n".join(under_lines) or "None"

    prompt = (
      f"Dataset statistics:\n{stats_text}"
      f"Top 5 happiest countries:\n{top_text}"
      f"Bottom 5 least happy countries:\n{bottom_text}"
      f"Factor correlations with happiness score:\n{corr_text}"
      f"Countries happier than their GDP predicts:\n{over_text}"
      f"Countries less happy than their GDP predicts:\n{under_text}"
      f"Question: {data.question}"
    )
    return PromptBuilderOutput(prompt=prompt, question=data.question)

# Step 2: sends the formatted prompt to SmolLLM and gets raw text response
class LLMRunner(Runnable[PromptBuilderOutput, LLMRunnerOutput]):
  name: str = "llm_runner"
  model_name: str = MODEL_NAME

  _pipe: Any = PrivateAttr(default=None)

  def invoke(self, data: PromptBuilderOutput) -> LLMRunnerOutput:
    if self._pipe is None:
      from transformers import pipeline
      self._pipe = pipeline(
        "text-generation",
        model=self.model_name,
        device="cpu"
      )
    
    # SmolLLM2-Instruct uses a chat format with roles.
    messages = [
      {
        # "system" sets the assistant's behaviour
        "role": "system",
        "content": (
          "You are a precise data analyst assistant."
          "Answer the user's question using only the dataset information provided."
          "Be concise and factual."
          "Always mention specific numbers from the data in your answer."
          "Answer in the same language as the question"
        ),
      },
      {
        # "user" is the actual prompt built earlier in PromptBuilder.
        "role": "user",
        "content": data.prompt,
      },
    ]
    result = self._pipe(
      messages,
      max_new_tokens=300,
      temperature=0.2,  # Keeps model focus on the data
      do_sample=True
    )

    raw_output = result[0]["generated_text"][-1]["content"]

    return LLMRunnerOutput(raw_output=raw_output, question=data.question)

# Step 3: Cleans the raw (model) output and returns a structured response.
class ResponseParser(Runnable[LLMRunnerOutput, ResponseParserOutput]):
  name: str = "response_parser"
  model_name: str = MODEL_NAME

  # Input: LLMRunnerOutput (ie., raw output + question)
  def invoke(self, data: LLMRunnerOutput) -> ResponseParserOutput:
    answer = data.raw_output.strip()

    for prefix in ["Assistant:", "Answer:", "Response:", "AI:"]:
      if answer.startswith(prefix):
        answer = answer[len(prefix):].strip()

    if answer and answer[-1] not in ".!?":
      last_stop = max(
        answer.rfind("."),
        answer.rfind("!"),
        answer.rfind("?"),
      )
      if last_stop > len(answer) // 2:
        answer = answer[:last_stop + 1]

    # If nothing is returned or nothing useful remains after cleaning, 
    # tell the user to try again
    if not answer:
      answer = "The model did not return a response. Try again."
    
    # Output: ResponseParserOutput (ie., clean answer + question + model name)
    return ResponseParserOutput(
      question=data.question,
      answer=answer,
      model=self.model_name,
    )
