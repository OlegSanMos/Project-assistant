"""Интеграционные тесты интерфейса: настоящий сервер Gradio и клиент gradio_client.

Запросы проходят тот же путь, что и из браузера: HTTP, очередь Gradio, обработчики событий,
состояние сессии. Вместо LLM используется сценарная модель из conftest.py.
"""

import socket

import pytest
from gradio_client import Client, handle_file

from conftest import SAMPLES, ScriptedChatModel, lab_responder
from labmentor.assistant import LabAssistant
from labmentor.config import Settings
from labmentor.ui import build_ui, launch_options

# режим, провайдер, модель, температура, окно модели, бюджет истории, резюме, top-k,
# интернет, число вопросов защиты, текущая работа
OPTIONS = ["auto", "openai", "gpt-4.1-mini", 0.3, 8192, 2000, True, 4, True, 3, None]
LAB_1 = "Лабораторная работа № 1. Проверка закона Ома для участка цепи"


def chat_text(chat: list) -> str:
    """Весь текст диалога, включая заголовки шагов агентов."""
    parts = []
    for message in chat:
        parts.append(str((message.get("metadata") or {}).get("title", "")))
        content = message["content"]
        for item in content if isinstance(content, list) else [content]:
            parts.append(item.get("text", "") if isinstance(item, dict) else str(item))
    return "\n".join(parts)


def has_image(chat: list) -> bool:
    return any(
        isinstance(item, dict) and item.get("type") == "file"
        for message in chat
        for item in (message["content"] if isinstance(message["content"], list) else [])
    )


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = Settings(
        provider="openai",
        openai_api_key="test-key",
        embeddings_provider="none",
        storage_dir=tmp_path_factory.mktemp("ui"),
    )
    llm = ScriptedChatModel(responder=lab_responder)
    demo = build_ui(LabAssistant(settings, llm_factory=lambda **kwargs: llm))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        quiet=True,
        **launch_options(settings),
    )
    yield Client(f"http://127.0.0.1:{port}/", verbose=False)
    demo.close()


def send(client, text, files=(), history=(), options=OPTIONS):
    """Отправляет сообщение и возвращает все промежуточные состояния вывода (потоковый ответ)."""
    message = {"text": text, "files": [handle_file(str(f)) for f in files]}
    job = client.submit(
        message, list(history), False, "ru-RU-SvetlanaNeural", *options, api_name="/on_message"
    )
    job.result()
    return job.outputs()


def test_rag_question_through_server(client):
    outputs = send(client, "Какова цель лабораторной работы № 1?")
    chat, _, _, context, *_ = outputs[-1]
    assert len(outputs) > 2  # ответ приходит потоком, а не одним блоком
    text = chat_text(chat)
    assert "Цель работы: проверить закон Ома" in text
    assert "🧭 Диспетчер" in text and "📚 Использованные фрагменты методичек" in text
    assert "Сообщений в окне:** 2 из 2" in context


def test_table_attachment_and_data_agent(client):
    outputs = send(
        client, "Построй график I(U) и найди сопротивление", files=[SAMPLES / "ohm_law.csv"]
    )
    chat = outputs[-1][0]
    text = chat_text(chat)
    assert "Загружена таблица ohm_law.csv: 14 строк" in text
    assert "🔧 Инструмент fit_dependency" in text and has_image(chat)
    # таблица из вложения сразу появляется на вкладке «Обработка данных»
    table, x_col, y_col = next(o[5:8] for o in outputs if "headers" in (o[5] or {}))
    assert table["headers"] == ["U_В", "I_мА"] and len(table["data"]) == 14
    assert (x_col["value"], y_col["value"]) == ("U_В", "I_мА")


def test_image_attachment_goes_to_vision(client):
    chat = send(client, "Определи постоянную времени", files=[SAMPLES / "rc_oscillogram.png"])[-1][
        0
    ]
    assert "tau около 1 с" in chat_text(chat) and has_image(chat)


def test_exam_dialog(client):
    history = []
    for answer in ("Проверь меня по работе 1", "Ток пропорционален напряжению", "Последовательно"):
        history = send(client, answer, history=history)[-1][0]
    text = chat_text(history)
    assert "Вопрос 1 из 2" in text and "8/10" in text and "Итог тренировочной защиты" in text


def test_data_tab_endpoints(client):
    table, x_col, y_col, stat, note = client.predict(
        handle_file(str(SAMPLES / "ohm_law.csv")), api_name="/on_table"
    )
    assert note.startswith("Загружена таблица") and stat["value"] is None
    table, *_, note = client.predict(
        "R, Ом = U_В / I_мА * 1000", x_col["value"], y_col["value"], api_name="/on_add_column"
    )
    assert note == "Добавлен столбец R_Ом" and "R_Ом" in table["headers"]
    plot, results = client.predict("U_В", "I_мА", 1, 0.95, "R_Ом", api_name="/on_analyze")
    assert plot.endswith(".png")
    assert "R² = 0.9999" in results and "Результат: R_Ом = 101.34 ± 0.32, P = 0.95" in results
    report, markdown = client.predict("Опыт при 22 °C", *OPTIONS[:-1], LAB_1, api_name="/on_report")
    assert report.endswith(".docx") and "## Выводы" in markdown


def test_context_window_controls(client):
    options = OPTIONS[:4] + [4096, 8000] + OPTIONS[6:]
    info, _ = client.predict(*options, api_name="/on_context_settings")
    # 4096 = промпт 500 + вопрос 300 + RAG 1200 + ответ 1500 + история 596
    assert "Окно модели 4096" in info and "история 596" in info and "из 596 токенов" in info
    chat, _, context, summary = client.predict(api_name="/on_clear")
    assert chat == [] and "Сообщений в окне:** 0 из 0" in context and summary == ""
