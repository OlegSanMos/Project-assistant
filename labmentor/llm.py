"""Фабрика языковых моделей и эмбеддингов.

Поддерживаются два провайдера:
* openai: любой OpenAI-совместимый API (OpenAI, ProxyAPI, OpenRouter, vLLM, LM Studio);
* ollama: открытые LLM, запущенные локально (Qwen2.5, Llama 3.1, Gemma 3 и др.).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings

PROVIDERS = ("openai", "ollama")


class LLMUnavailableError(RuntimeError):
    """Модель не настроена или недоступна."""


def create_chat_model(
    settings: Settings,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
) -> BaseChatModel:
    """Создает чат-модель LangChain для выбранного провайдера."""
    provider = (provider or settings.provider).lower()
    model = model or settings.chat_model
    temperature = settings.temperature if temperature is None else temperature

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=num_ctx or settings.num_ctx,
            num_predict=settings.max_answer_tokens,
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMUnavailableError(
                "Не задан OPENAI_API_KEY. Укажите ключ в файле .env или выберите провайдера ollama."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            temperature=temperature,
            max_tokens=settings.max_answer_tokens,
            timeout=120,
            max_retries=2,
        )
    raise LLMUnavailableError(f"Неизвестный провайдер LLM: {provider}")


def create_embeddings(settings: Settings) -> Embeddings | None:
    """Модель эмбеддингов для RAG или None (тогда используется только поиск BM25)."""
    provider = settings.embeddings_provider
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        inner: Embeddings = OllamaEmbeddings(
            model=settings.embedding_model, base_url=settings.ollama_base_url
        )
    elif provider == "openai" and settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        inner = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            check_embedding_ctx_length=False,
        )
    else:
        return None
    cache_file = (
        settings.cache_dir / f"embeddings_{provider}_{_slug(settings.embedding_model)}.json"
    )
    return CachedEmbeddings(inner, cache_file)


class CachedEmbeddings(Embeddings):
    """Эмбеддинги с кэшем на диске: при повторном запуске векторы не пересчитываются."""

    def __init__(self, inner: Embeddings, cache_file: Path) -> None:
        self.inner = inner
        self.cache_file = cache_file
        self._cache: dict[str, list[float]] = {}
        if cache_file.exists():
            self._cache = json.loads(cache_file.read_text(encoding="utf-8"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha1(text.encode("utf-8")).hexdigest() for text in texts]
        missing = [
            (key, text) for key, text in zip(keys, texts, strict=True) if key not in self._cache
        ]
        if missing:
            vectors = self.inner.embed_documents([text for _, text in missing])
            self._cache.update(
                {key: vector for (key, _), vector in zip(missing, vectors, strict=True)}
            )
            self.cache_file.write_text(json.dumps(self._cache), encoding="utf-8")
        return [self._cache[key] for key in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)
