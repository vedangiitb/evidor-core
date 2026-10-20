"""Provider-neutral data structures and models."""

from .generation import (
    GenerationRequest,
    GenerationResponse,
    Message,
    MessageRole,
    SearchResult,
    ToolCall,
)
from .streaming import (
    AsyncStreamResponse,
    StreamChunk,
    StreamResponse,
)
from .thinking import (
    AnthropicThinkingConfig,
    AnyThinkingConfig,
    GeminiThinkingConfig,
    OpenAIThinkingConfig,
    ThinkingConfig,
    ThinkingEffort,
)

__all__ = [
    "ThinkingConfig",
    "ThinkingEffort",
    "OpenAIThinkingConfig",
    "AnthropicThinkingConfig",
    "GeminiThinkingConfig",
    "AnyThinkingConfig",
    "SearchResult",
    "ToolCall",
    "MessageRole",
    "Message",
    "GenerationRequest",
    "GenerationResponse",
    "StreamChunk",
    "StreamResponse",
    "AsyncStreamResponse",
]

for _name in __all__:
    _item = globals().get(_name)
    if isinstance(_item, type) and getattr(_item, "__module__", None) != "builtins":
        try:
            _item.__module__ = "evidor.models"
        except (TypeError, AttributeError):
            pass

