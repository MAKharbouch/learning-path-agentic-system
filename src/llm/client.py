"""LLM chat model factory with multi-provider support.

Returns a LangChain BaseChatModel based on LLM_PROVIDER and LLM_MODEL env vars.
Any provider supported by langchain's init_chat_model can be used.
"""

from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel


def get_chat_model() -> BaseChatModel:
    """Return a configured LangChain chat model for the active provider.

    Reads LLM_PROVIDER and LLM_MODEL from config and delegates to
    langchain's init_chat_model. Any provider that LangChain supports
    can be used — just set LLM_PROVIDER and LLM_MODEL in .env.

    Returns:
        A BaseChatModel instance.
    """
    from config import LLM_MODEL, LLM_PROVIDER

    return cast(
        BaseChatModel,
        init_chat_model(
            LLM_MODEL or None,
            model_provider=LLM_PROVIDER.lower(),
            temperature=0,
        ),
    )
