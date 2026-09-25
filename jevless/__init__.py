"""jevless: Jev-style typed decisions (choice, noul, score) from any model whose API exposes first-token logprobs."""
from .backends import BackendError, LlamaCpp, LMStudio, OpenAIChat, Transformers
from .core import Answer, Choice, Decider, DecisionError, Noul, Score, fit_temperature

__version__ = "0.1.0"
__all__ = ["Decider", "Noul", "Choice", "Score", "Answer", "DecisionError", "fit_temperature",
           "OpenAIChat", "LMStudio", "LlamaCpp", "Transformers", "BackendError"]
