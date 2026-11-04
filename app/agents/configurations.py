from typing import Any

from app.adapters.langgraph_adapter import LangGraphAdapter
from app.agents.langgraph_agent import (
    BASELINE_SYSTEM_PROMPT,
    HARDENED_SYSTEM_PROMPT,
    LangGraphAgent,
)
from app.agents.llm_client import (
    DEFAULT_NVIDIA_MODEL,
    NvidiaClientConfig,
    get_llm,
)
from app.observability.logger import EventLogger
from app.tools.mock_tools import ControlledToolExecutor, ToolInterceptor


class LangGraphBaselineAgent(LangGraphAdapter):
    """
    Real LLM-powered Baseline Agent using the LangGraph StateGraph workflow.

    Characteristics:
    - Basic system instructions without explicit instruction hierarchy.
    - Controlled tools without hardened security middleware.
    - No security boundary wrapping for untrusted external content.
    - Records tool usage without security interception.
    """

    def __init__(
        self,
        llm: Any = None,
        config: NvidiaClientConfig | None = None,
        event_logger: EventLogger | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 5,
        interceptor: Any = None,
    ):
        self.config = config
        self.logger = event_logger or EventLogger()
        self.agent_type = "baseline"
        self.logger.agent_type = "baseline"
        self.tool_executor = interceptor or ControlledToolExecutor(event_logger=self.logger)

        resolved_llm = llm
        if resolved_llm is None and self.config is not None:
            try:
                resolved_llm = get_llm(config=self.config)
            except Exception:
                resolved_llm = None

        agent = LangGraphAgent(
            llm=resolved_llm,
            config=self.config,
            event_logger=self.logger,
            system_prompt=system_prompt or BASELINE_SYSTEM_PROMPT,
            max_iterations=max_iterations,
            interceptor=self.tool_executor,
            separate_untrusted_context=False,
        )

        super().__init__(agent=agent)

    @property
    def model_name(self) -> str:
        return self.agent.model_name

    @property
    def model_config(self) -> dict[str, Any]:
        return self.agent.model_config


class LangGraphHardenedAgent(LangGraphAdapter):
    """
    Real LLM-powered Hardened Agent using the LangGraph StateGraph workflow.

    Characteristics:
    - Applies explicit instruction hierarchy with highest priority given to system rules.
    - Isolates untrusted external content in <untrusted_external_content> boundaries.
    - Enforces tool allowlisting via ToolInterceptor.
    - Validates tool arguments and blocks path traversal attempts.
    - Records all security enforcement decisions (permission_denied, argument_validation_failed).
    """

    def __init__(
        self,
        llm: Any = None,
        config: NvidiaClientConfig | None = None,
        event_logger: EventLogger | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 5,
        interceptor: ToolInterceptor | None = None,
    ):
        self.config = config
        self.logger = event_logger or EventLogger()
        self.agent_type = "hardened"
        self.logger.agent_type = "hardened"
        self.interceptor = interceptor or ToolInterceptor(event_logger=self.logger)

        resolved_llm = llm
        if resolved_llm is None and self.config is not None:
            try:
                resolved_llm = get_llm(config=self.config)
            except Exception:
                resolved_llm = None

        agent = LangGraphAgent(
            llm=resolved_llm,
            config=self.config,
            event_logger=self.logger,
            system_prompt=system_prompt or HARDENED_SYSTEM_PROMPT,
            max_iterations=max_iterations,
            interceptor=self.interceptor,
            separate_untrusted_context=True,
        )

        super().__init__(agent=agent)

    @property
    def model_name(self) -> str:
        return self.agent.model_name

    @property
    def model_config(self) -> dict[str, Any]:
        return self.agent.model_config


def create_agent_pair(
    llm: Any = None,
    config: NvidiaClientConfig | None = None,
    baseline_logger: EventLogger | None = None,
    hardened_logger: EventLogger | None = None,
    max_iterations: int = 5,
    mode: str | None = None,
) -> tuple[LangGraphBaselineAgent, LangGraphHardenedAgent]:
    """
    Creates a paired Baseline and Hardened agent using the exact same LLM
    and inference configuration for fair comparative evaluation.

    Execution modes:
    - 'mock': Uses deterministic AdversarialScenarioLLM for offline evaluation.
    - 'real': Uses live NVIDIA NIM ChatNVIDIA inference loaded from environment.
              Does NOT silently fall back to mock mode if inference fails.
    """
    shared_config = config
    shared_llm = llm

    if shared_llm is None:
        if mode == "real":
            # Real mode: explicit live NVIDIA inference
            if shared_config is None:
                shared_config = NvidiaClientConfig.from_env()
            shared_llm = get_llm(config=shared_config)
        else:
            # Mock mode: deterministic adversarial scenario LLM
            from app.agents.mock_llm import AdversarialScenarioLLM
            shared_llm = AdversarialScenarioLLM()
            if shared_config is None:
                shared_config = NvidiaClientConfig(
                    api_key="mock_sentinel_key",
                    model=getattr(shared_llm, "model_name", DEFAULT_NVIDIA_MODEL)
                )
    else:
        if shared_config is None:
            model_name = getattr(shared_llm, "model_name", getattr(shared_llm, "model", DEFAULT_NVIDIA_MODEL))
            shared_config = NvidiaClientConfig(
                api_key="sentinel_shared_key",
                model=str(model_name)
            )

    b_logger = baseline_logger or EventLogger()
    h_logger = hardened_logger or EventLogger()

    baseline_agent = LangGraphBaselineAgent(
        llm=shared_llm,
        config=shared_config,
        event_logger=b_logger,
        max_iterations=max_iterations,
    )

    hardened_agent = LangGraphHardenedAgent(
        llm=shared_llm,
        config=shared_config,
        event_logger=h_logger,
        max_iterations=max_iterations,
    )

    return baseline_agent, hardened_agent

