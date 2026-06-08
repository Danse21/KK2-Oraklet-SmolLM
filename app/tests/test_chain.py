# Test for each Runnable step
# with each test given a step known input and verifies the output is correct

# Run with: uv run pytest app/tests/test_chain.py -v

from app.chain.steps import PromptBuilder, ResponseParser
from app.schemas import PromptBuilderInput, LLMRunnerOutput

# stats dict that simulates what df.describe().to_dict() returns
sample_stats = {
  "score": {"mean": 5.4, "std": 1.1, "min": 2.8, "max": 7.7},
  "gdp": {"mean": 0.9, "std": 0.4, "min": 0.0, "max": 1.7},
}

# Top and bottom that simulates what get_top_bottom() returns
sample_top = [
  {"country": "Finland",     "happiness_score": 7.769},
  {"country": "Denmark",     "happiness_score": 7.600},
  {"country": "Norway",      "happiness_score": 7.554},
  {"country": "Iceland",     "happiness_score": 7.494},
  {"country": "Netherlands", "happiness_score": 7.488},
]
sample_bottom = [
  {"country": "South Sudan",             "happiness_score": 2.853},
  {"country": "Central African Republic","happiness_score": 3.083},
  {"country": "Afghanistan",             "happiness_score": 3.203},
  {"country": "Tanzania",                "happiness_score": 3.231},
  {"country": "Rwanda",                  "happiness_score": 3.334},
]

# Correlations that simulates what get_correlations() returns
sample_correlations = {
  "gdp":            0.794,
  "life_expectancy":0.780,
  "social_support": 0.777,
  "freedom":        0.567,
  "corruption":     0.386,
  "generosity":     0.076,
}

# Outliers that simulates what get_outliers() returns.
sample_outliers = {
  "over_performers":  [{"country": "Costa Rica", "happiness_score": 7.167}],
  "under_performers": [{"country": "Singapore",  "happiness_score": 6.343}],
}

sample_question = "Which country has the highest happiness score?"

# Builds a PromptBuilderInput with all required fields.
def _make_input(question=sample_question):
  return PromptBuilderInput(
    question=question,
    stats=sample_stats,
    top=sample_top,
    bottom=sample_bottom,
    correlations=sample_correlations,
    outliers=sample_outliers,
  )

def test_prompt_builder_contains_the_question():
  """PromptBuilder must include the user's question in the prompt"""
  output = PromptBuilder().invoke(_make_input())
  assert sample_question in output.prompt

def test_prompt_builder_contains_stats():
  """PromptBuilder must include the dataset statistics in the prompt"""
  output = PromptBuilder().invoke(_make_input())
  assert "score" in output.prompt

def test_prompt_builder_contains_top_country():
  """PromptBuilder includes the top-ranked countries in the prompt."""
  output = PromptBuilder().invoke(_make_input())
  assert "Finland" in output.prompt

def test_prompt_builder_contains_bottom_country():
  """PromptBuilder includes the bottom-ranked countries in the prompt."""
  output = PromptBuilder().invoke(_make_input())
  assert "South Sudan" in output.prompt

def test_prompt_builder_contains_correlations():
  """PromptBuilder includes correlation data in the prompt."""
  output = PromptBuilder().invoke(_make_input())
  assert "gdp" in output.prompt

def test_prompt_builder_contains_outliers():
  """PromptBuilder includes outlier countries in the prompt."""
  output = PromptBuilder().invoke(_make_input())
  assert "Costa Rica" in output.prompt

def test_prompt_builder_carries_question_forward():
  """PromptBuilderOutput carries the original question through unchanged."""
  output = PromptBuilder().invoke(_make_input())
  assert output.question == sample_question

def test_response_parser_strips_whitespace():
  """ResponseParser strips leading and trailing whitespace from the model output."""
  raw_input = LLMRunnerOutput(
    raw_output="Finland has the highest score. \n",
    question=sample_question,
  )
  output = ResponseParser().invoke(raw_input)
  assert output.answer == "Finland has the highest score."

def test_response_parser_handles_empty_output():
  """responseParserOutput returns a fallback message when the model returns nothing."""
  raw_input = LLMRunnerOutput(
    raw_output=" ",
    question=sample_question,
  )
  output = ResponseParser().invoke(raw_input)
  assert len(output.answer) > 0

def test_response_carries_question_and_model_forward():
  """ResponseParserOutput includes the original question and the model name."""
  raw_input = LLMRunnerOutput(
    raw_output="Finland is the happiest country.",
    question=sample_question,
  )
  output = ResponseParser().invoke(raw_input)
  assert output.question == sample_question
  assert output.model == "llama-3.1-8b-instant"

def test_response_parser_strips_common_prefixes():
  """ResponseParser removes prefixes like 'Answer:' and 'Assistant:' from the reply."""
  parser= ResponseParser()
  for prefix in ["Answer:", "Assistant:", "Response:", "AI:"]:
    raw_input = LLMRunnerOutput(
    raw_output=f"{prefix} Finland is the happiest country.",
    question=sample_question,
  )
    output = parser.invoke(raw_input)
    assert not output.answer.startswith(prefix)
    assert "Finland" in output.answer

def test_response_parser_trims_incomplete_sentence():
  """ResponseParser trims a cut-off reply back to the last complete sentence."""
  raw_input = LLMRunnerOutput(
    raw_output="GDP per capita is the strongest predictor of happiness. Denmark is second with 7",
    question=sample_question,
  )
  output = ResponseParser().invoke(raw_input)
  assert output.answer[-1] in ".!?"
  assert "GDP per capita" in output.answer
