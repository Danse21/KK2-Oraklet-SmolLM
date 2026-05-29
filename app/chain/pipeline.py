
from app.chain.steps import PromptBuilder, LLMRunner, ResponseParser

# Assembles the three chain steps into one pipeline using the | operator
# Each | creates a RunnableSequence that connects two steps
# call oraklet.invoke(prompt_builder_input) to run all the 3 steps automatically in sequence

oraklet = PromptBuilder() | LLMRunner ()| ResponseParser ()


#   PromptBuilder  — takes the question + stats, builds a prompt
#   LLMRunner      — sends the prompt to SmolLLM, gets raw output back
#   ResponseParser — cleans the raw output into a structured response