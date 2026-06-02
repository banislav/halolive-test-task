from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from deep_agents.config import DeepAgentsSettings
from deep_agents.langchain.models import build_chat_model
from deep_agents.langchain.prompts import (
    build_content_reasoning_messages,
    build_prompt_classifier_messages,
)
from deep_agents.models import (
    PromptClassification,
    PromptQueueItem,
    PromptReasoningInput,
    PromptResponse,
)


def build_prompt_classifier(
    model: BaseChatModel | None = None,
    settings: DeepAgentsSettings | None = None,
) -> Runnable[PromptQueueItem, PromptClassification]:
    """Build a LangChain runnable that classifies queued user prompts."""
    chat_model = model or build_chat_model(settings)
    structured_model = chat_model.with_structured_output(PromptClassification)
    return RunnableLambda(_coerce_prompt_item) | RunnableLambda(
        build_prompt_classifier_messages
    ) | structured_model | RunnableLambda(_coerce_prompt_classification)


def build_content_reasoning_agent(
    model: BaseChatModel | None = None,
    settings: DeepAgentsSettings | None = None,
) -> Runnable[PromptReasoningInput, PromptResponse]:
    """Build a LangChain runnable that answers read-only queued prompts."""
    chat_model = model or build_chat_model(settings)
    structured_model = chat_model.with_structured_output(PromptResponse)
    return RunnableLambda(_coerce_reasoning_input) | RunnableLambda(
        build_content_reasoning_messages
    ) | structured_model | RunnableLambda(_coerce_prompt_response)


def _coerce_prompt_item(value: PromptQueueItem | dict[str, Any]) -> PromptQueueItem:
    if isinstance(value, PromptQueueItem):
        return value
    return PromptQueueItem(**value)


def _coerce_reasoning_input(
    value: PromptReasoningInput | dict[str, Any],
) -> PromptReasoningInput:
    if isinstance(value, PromptReasoningInput):
        return value
    return PromptReasoningInput(**value)


def _coerce_prompt_classification(
    value: PromptClassification | dict[str, Any],
) -> PromptClassification:
    if isinstance(value, PromptClassification):
        return value
    return PromptClassification(**value)


def _coerce_prompt_response(value: PromptResponse | dict[str, Any]) -> PromptResponse:
    if isinstance(value, PromptResponse):
        return value
    return PromptResponse(**value)
