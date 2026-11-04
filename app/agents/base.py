from abc import ABC, abstractmethod
from typing import Any

from app.core.models import AgentResponse


class AgentAdapter(ABC):

    @abstractmethod
    def run(
        self,
        task: str,
        untrusted_context: str | None = None,
        tools: Any = None,
    ) -> AgentResponse:
        pass
