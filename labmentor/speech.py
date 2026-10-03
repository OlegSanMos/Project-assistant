"""Речевая модальность: распознавание (STT) и синтез (TTS) речи.

Распознавание: OpenAI-совместимый Whisper API или локальная открытая модель
Whisper (faster-whisper). Синтез: нейросетевые голоса Edge TTS, резервный
вариант Google TTS (gTTS).
"""

from __future__ import annotations

import re
import uuid
from functools import lru_cache
from pathlib import Path

from .config import Settings

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"}
VOICES = {
    "Светлана (женский)": "ru-RU-SvetlanaNeural",
    "Дмитрий (мужской)": "ru-RU-DmitryNeural",
}
MAX_TTS_CHARS = 1500


def transcribe(path: str | Path, settings: Settings) -> str:
    """Распознает русскую речь из аудиофайла и возвращает текст."""
    engine = settings.stt_engine
    if engine == "auto":
        engine = "openai" if settings.provider == "openai" and settings.openai_api_key else "local"
    if engine == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None)
        with open(path, "rb") as audio:
            result = client.audio.transcriptions.create(
                model=settings.stt_model, file=audio, language="ru"
            )
        return result.text.strip()
    segments, _ = _whisper(settings.whisper_size).transcribe(
        str(path), language="ru", vad_filter=True
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


@lru_cache(maxsize=2)
def _whisper(size: str):
    """Локальная модель Whisper (загружается один раз при первом обращении)."""
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def text_for_speech(text: str) -> str:
    """Убирает разметку Markdown, ссылки и формулы, которые плохо звучат при озвучивании."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[[^\]]*\.(md|pdf|docx|txt)\]", " ", text)
    text = re.sub(r"[#*_`>|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_TTS_CHARS]


def synthesize(text: str, voice: str, out_dir: Path) -> str | None:
    """Озвучивает текст и возвращает путь к mp3-файлу (или None, если синтез недоступен)."""
    clean = text_for_speech(text)
    if not clean:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"answer_{uuid.uuid4().hex[:8]}.mp3"
    try:
        import edge_tts

        edge_tts.Communicate(clean, voice).save_sync(str(path))
        return str(path)
    except Exception:
        pass
    try:
        from gtts import gTTS

        gTTS(clean, lang="ru").save(str(path))
        return str(path)
    except Exception:
        return None
