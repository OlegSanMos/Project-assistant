"""Тесты речи, инструментов интернета, фабрики моделей и настроек.

Внешние сервисы (Whisper, Edge TTS, gTTS, DuckDuckGo, Википедия) подменяются заглушками,
поэтому тесты проверяют логику модулей без доступа к сети.
"""

from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from labmentor import speech, tools
from labmentor.config import Settings
from labmentor.llm import (
    CachedEmbeddings,
    LLMUnavailableError,
    create_chat_model,
    create_embeddings,
)


# ---------- речь ----------
def test_text_for_speech_removes_markup():
    text = "## Ответ\n**R = 101 Ом** [lr1_ohm_law.md], см. https://example.com и `код`"
    assert speech.text_for_speech(text) == "Ответ R = 101 Ом , см. и код"


def test_local_transcription_uses_whisper(monkeypatch, settings, tmp_path):
    class FakeWhisper:
        def transcribe(self, path, language, vad_filter):
            assert language == "ru" and vad_filter
            return [SimpleNamespace(text=" Какова цель "), SimpleNamespace(text="работы? ")], None

    monkeypatch.setattr(speech, "_whisper", lambda size: FakeWhisper())
    settings.stt_engine = "local"
    audio = tmp_path / "question.wav"
    audio.write_bytes(b"RIFF")
    assert speech.transcribe(audio, settings) == "Какова цель работы?"


def test_api_transcription_uses_openai_client(monkeypatch, settings, tmp_path):
    calls = {}

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            calls["key"] = api_key
            create = lambda model, file, language: SimpleNamespace(text=" Закон Ома ")  # noqa: E731
            self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=create))

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    audio = tmp_path / "question.webm"
    audio.write_bytes(b"data")
    assert speech.transcribe(audio, settings) == "Закон Ома"  # stt_engine=auto, есть ключ API
    assert calls["key"] == "test-key"


def test_synthesis_falls_back_from_edge_to_gtts(monkeypatch, tmp_path):
    class Edge:
        def __init__(self, text, voice):
            self.text = text

        def save_sync(self, path):
            open(path, "wb").write(b"edge")

    class BrokenEdge(Edge):
        def save_sync(self, path):
            raise ConnectionError("нет сети")

    class GoogleTTS:
        def __init__(self, text, lang):
            assert lang == "ru"

        def save(self, path):
            open(path, "wb").write(b"gtts")

    monkeypatch.setattr("edge_tts.Communicate", Edge)
    path = speech.synthesize("**Ответ:** R = 101 Ом", "ru-RU-SvetlanaNeural", tmp_path)
    assert open(path, "rb").read() == b"edge"

    monkeypatch.setattr("edge_tts.Communicate", BrokenEdge)
    monkeypatch.setattr("gtts.gTTS", GoogleTTS)
    path = speech.synthesize("Ответ", "ru-RU-SvetlanaNeural", tmp_path)
    assert open(path, "rb").read() == b"gtts"

    monkeypatch.setattr("gtts.gTTS", BrokenEdge)
    assert speech.synthesize("Ответ", "ru-RU-SvetlanaNeural", tmp_path) is None
    assert speech.synthesize("```код```", "ru-RU-SvetlanaNeural", tmp_path) is None


# ---------- инструменты интернета ----------
def test_web_search_formats_results(monkeypatch):
    class FakeDDGS:
        def text(self, query, region, max_results):
            assert region == "ru-ru"
            return [{"title": "Сталь 45", "href": "https://example.com/45", "body": "E = 200 ГПа"}]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    result = tools.web_search.invoke({"query": "модуль Юнга стали 45"})
    assert "1. Сталь 45" in result and "E = 200 ГПа" in result


def test_web_search_reports_network_error(monkeypatch):
    class OfflineDDGS:
        def text(self, *args, **kwargs):
            raise ConnectionError("нет сети")

    monkeypatch.setattr("ddgs.DDGS", OfflineDDGS)
    assert "недоступен" in tools.web_search.invoke({"query": "плотность меди"})


def test_wikipedia_search(monkeypatch):
    page = {
        "title": "Закон Гука",
        "index": 1,
        "fullurl": "https://ru.wikipedia.org/wiki/Закон_Гука",
        "extract": "Утверждение о линейной связи деформации и напряжения.",
    }
    response = SimpleNamespace(json=lambda: {"query": {"pages": {"1": page}}})
    monkeypatch.setattr(tools.requests, "get", lambda *args, **kwargs: response)
    result = tools.wikipedia_search.invoke({"query": "закон Гука"})
    assert result.startswith("Закон Гука (https://ru.wikipedia.org/wiki/Закон_Гука)")


def test_calculator_tool():
    assert tools.calculator.invoke({"expression": "sqrt(0.5 * 100)"}) == "sqrt(0.5 * 100) = 7.07107"
    assert tools.calculator.invoke({"expression": "import os"}).startswith("Ошибка")


# ---------- модели, эмбеддинги, настройки ----------
def test_chat_model_factory(settings):
    openai_model = create_chat_model(settings, model="gpt-4.1-mini", temperature=0)
    assert isinstance(openai_model, ChatOpenAI) and openai_model.max_tokens == 1500
    ollama_model = create_chat_model(settings, provider="ollama", model="qwen2.5:7b", num_ctx=4096)
    assert isinstance(ollama_model, ChatOllama) and ollama_model.num_ctx == 4096
    settings.openai_api_key = ""
    with pytest.raises(LLMUnavailableError):
        create_chat_model(settings)


def test_embeddings_factory_and_cache(settings, tmp_path):
    assert create_embeddings(settings) is None  # EMBEDDINGS_PROVIDER=none: только BM25
    settings.embeddings_provider = "ollama"
    assert isinstance(create_embeddings(settings), CachedEmbeddings)

    class Counting(Embeddings):
        calls = 0

        def embed_documents(self, texts):
            Counting.calls += 1
            return [[float(len(t)), 1.0] for t in texts]

        def embed_query(self, text):
            return [float(len(text)), 1.0]

    cache = tmp_path / "cache.json"
    first = CachedEmbeddings(Counting(), cache)
    assert first.embed_documents(["закон Ома", "ВАХ"]) == [[9.0, 1.0], [3.0, 1.0]]
    first.embed_documents(["ВАХ"])
    second = CachedEmbeddings(Counting(), cache)  # новый запуск читает кэш с диска
    second.embed_documents(["закон Ома"])
    assert Counting.calls == 1


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("NUM_CTX", "4096")
    monkeypatch.setenv("HISTORY_TOKENS", "1500")
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_PROVIDER", raising=False)
    settings = Settings.from_env()
    assert (settings.chat_model, settings.vision_model, settings.embedding_model) == (
        "qwen2.5:7b",
        "qwen2.5vl:7b",
        "bge-m3",
    )
    assert settings.num_ctx == 4096 and settings.history_tokens == 1500
    assert settings.embeddings_provider == "ollama"


def test_check_setup_gives_hints(monkeypatch, settings):
    from labmentor.llm import check_setup

    assert check_setup(settings) == []  # ключ задан, эмбеддинги отключены
    settings.openai_api_key = ""
    assert "OPENAI_API_KEY" in check_setup(settings)[0]

    ollama = Settings(provider="ollama", storage_dir=settings.storage_dir)
    tags = {"models": [{"name": "qwen2.5:7b"}, {"name": "bge-m3:latest"}]}
    monkeypatch.setattr(
        "labmentor.llm.requests.get", lambda *args, **kwargs: SimpleNamespace(json=lambda: tags)
    )
    assert check_setup(ollama) == [
        "Модель qwen2.5vl:7b не загружена: выполните ollama pull qwen2.5vl:7b"
    ]

    def offline(*args, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr("labmentor.llm.requests.get", offline)
    assert "запустите Ollama" in check_setup(ollama)[0]
