"""Управление контекстным окном диалога.

Полная история хранится в списке сообщений LangChain. В промпт попадает только
окно последних сообщений, которое укладывается в бюджет токенов (trim_messages).
Сообщения, вытесненные из окна, не теряются: цепочка суммаризации сжимает их
в краткое резюме, и оно передается модели в системном промпте.
"""

from __future__ import annotations

from functools import partial

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import Runnable

# Приближенный подсчет токенов: для русского текста около 3 символов на токен.
count_tokens = partial(count_tokens_approximately, chars_per_token=3.0)


def plan_context(
    num_ctx: int, history_tokens: int, top_k: int, chunk_size: int, answer_tokens: int
) -> dict:
    """Распределяет контекстное окно модели между частями промпта и ответом.

    Окно = системный промпт + фрагменты RAG + история диалога + вопрос + ответ.
    История получает запрошенный бюджет, но не больше остатка окна.
    """
    system, question = 500, 300
    rag = top_k * chunk_size // 3
    free = num_ctx - system - question - rag - answer_tokens
    return {
        "num_ctx": num_ctx,
        "system": system,
        "rag": rag,
        "answer": answer_tokens,
        "history": max(256, min(history_tokens, free)),
    }


def format_dialog(messages: list[BaseMessage]) -> str:
    """Преобразует сообщения в текст вида «Студент: ... / Ассистент: ...»."""
    names = {"human": "Студент", "ai": "Ассистент"}
    return "\n".join(f"{names.get(m.type, m.type)}: {m.text}" for m in messages)


class ConversationMemory:
    """Скользящее окно истории с резюме вытесненной части диалога."""

    def __init__(self, max_tokens: int = 2000, use_summary: bool = True) -> None:
        self.messages: list[BaseMessage] = []  # полная история диалога
        self.max_tokens = max_tokens
        self.use_summary = use_summary
        self.summary = ""
        self._summarized = 0  # сколько вытесненных сообщений уже учтено в резюме

    def add_turn(self, question: str, answer: str) -> None:
        """Сохраняет пару «вопрос студента / ответ ассистента»."""
        self.messages += [HumanMessage(content=question), AIMessage(content=answer)]

    def window(self) -> list[BaseMessage]:
        """Последние сообщения, укладывающиеся в бюджет max_tokens."""
        return trim_messages(
            self.messages,
            max_tokens=self.max_tokens,
            token_counter=count_tokens,
            strategy="last",
            start_on="human",
            allow_partial=False,
        )

    def evicted(self) -> list[BaseMessage]:
        """Сообщения, которые уже не помещаются в окно."""
        return self.messages[: len(self.messages) - len(self.window())]

    def summary_text(self) -> str:
        """Резюме для системного промпта (если суммаризация включена)."""
        return self.summary if self.use_summary and self.summary else "нет"

    def update_summary(self, summary_chain: Runnable) -> bool:
        """Добавляет в резюме новые вытесненные сообщения. True, если резюме изменилось."""
        if not self.use_summary:
            return False
        evicted = self.evicted()
        new_messages = evicted[self._summarized :]
        if not new_messages:
            return False
        self.summary = summary_chain.invoke(
            {"summary": self.summary or "пока пусто", "dialog": format_dialog(new_messages)}
        ).strip()
        self._summarized = len(evicted)
        return True

    def stats(self) -> dict:
        """Показатели заполнения контекстного окна для интерфейса."""
        window = self.window()
        return {
            "tokens": count_tokens(window),
            "limit": self.max_tokens,
            "in_window": len(window),
            "total": len(self.messages),
            "summary_words": len(self.summary.split()) if self.use_summary else 0,
        }

    def clear(self) -> None:
        self.messages = []
        self.summary = ""
        self._summarized = 0
