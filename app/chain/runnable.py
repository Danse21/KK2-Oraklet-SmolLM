from pydantic import BaseModel, ConfigDict, SerializeAsAny
from typing import Any, Callable, Generic, TypeVar

# TypeVars are placeholders for types
I = TypeVar("I")
O = TypeVar("O")
M = TypeVar("M")

# this class knows what type goes in (I) and what comes out (O)
# through BaseModel, pydantic handles validation and serialization automatically
class Runnable(BaseModel, Generic[I, O]):
  model_config = ConfigDict(arbitrary_types_allowed=True) # deals with objects that are not known by default

  name: str | None = None
  def invoke(self, data: I) -> O:

    raise NotImplementedError("Subclass must implement invoke()")

  def __or__(self, other: Any) -> "RunnableSequence": # this makes the | operator to work
    if isinstance(other, Runnable):
    # For example when you write PromptBuilder() | LLMRunner(),
    # __or__ function returns a RunnableSequence that remembers both steps
    # chain two Runnable steps together
      return RunnableSequence.model_construct(
      first=self,
      second=other,
    )
    # wrap chained function in a RunnableLambda automatically
    if callable(other):
      return RunnableSequence.model_construct(
        first=self,
        second=RunnableLambda.model_construct(func=other, name=other.__name__),
        name=other.__name__,
      )
    return NotImplemented

  # Handles the case where a plain function is on the LEFT side of |
  def __ror__(self, other: Any) -> Any:
    if callable(other):
      return RunnableSequence.model_construct(
        first=RunnableLambda.model_construct(func=other),
        second=self,
        name=other.__name__,
      )
    return NotImplemented

# Wraps a function so it can participate in a chain
class RunnableLambda(Runnable[I, O]):
  func: Callable[[I], O]

  def invoke(self, data: I) -> O:
    return self.func(data)  # calls the wrapped function

# Holds two steps and runs them in sequence
class RunnableSequence(Runnable[I, O], Generic[I, M, O]):
  first: SerializeAsAny[Runnable[I, M]] # runs first
  second: SerializeAsAny[Runnable[M, O]]  # runs second after the first run finishes.
                                          # M connects them: output of first = input of second
  
  def invoke(self, data: I) -> O:
    # Run first, take its output, pass it directly into second
    intermediate = self.first.invoke(data)
    return self.second.invoke(intermediate)
