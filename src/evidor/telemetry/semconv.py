"""Standard semantic attribute names for Evidor telemetry.

These constants follow OpenTelemetry GenAI Semantic Conventions and
OpenInference conventions where applicable, while defining Evidor-specific
attributes under the `evidor.*` namespace.
"""

from typing import Final

# --- OpenTelemetry GenAI Semantic Conventions ---
GEN_AI_SYSTEM: Final[str] = "gen_ai.system"
GEN_AI_REQUEST_MODEL: Final[str] = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL: Final[str] = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS: Final[str] = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: Final[str] = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_TOTAL_TOKENS: Final[str] = "gen_ai.usage.total_tokens"

# --- OpenInference Semantic Conventions ---
OPENINFERENCE_SPAN_KIND: Final[str] = "openinference.span.kind"
SPAN_KIND_AGENT: Final[str] = "AGENT"
SPAN_KIND_CHAIN: Final[str] = "CHAIN"
SPAN_KIND_LLM: Final[str] = "LLM"
SPAN_KIND_TOOL: Final[str] = "TOOL"

# --- General OpenTelemetry Conventions ---
ERROR_TYPE: Final[str] = "error.type"
ERROR_MESSAGE: Final[str] = "error.message"

# --- Evidor Specific Semantic Conventions ---
EVIDOR_AGENT_RUN_ID: Final[str] = "evidor.agent.run_id"
EVIDOR_AGENT_ITERATIONS: Final[str] = "evidor.agent.iterations"
EVIDOR_TOOL_NAME: Final[str] = "evidor.tool.name"
EVIDOR_TOOL_CALL_ID: Final[str] = "evidor.tool.call_id"
EVIDOR_MCP_SERVER_NAME: Final[str] = "evidor.mcp.server_name"
EVIDOR_RETRY_COUNT: Final[str] = "evidor.retry.count"
EVIDOR_RETRY_OUTCOME: Final[str] = "evidor.retry.outcome"
EVIDOR_RETRY_DELAY_SECONDS: Final[str] = "evidor.retry.delay_seconds"
EVIDOR_RETRY_MAX: Final[str] = "evidor.retry.max"

