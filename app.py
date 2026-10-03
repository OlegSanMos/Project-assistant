"""Запуск веб-приложения ЛабМентор.

Пример: python app.py --host 0.0.0.0 --port 7860
"""

import argparse

from labmentor.assistant import LabAssistant
from labmentor.config import Settings
from labmentor.llm import check_setup
from labmentor.ui import build_ui, launch_options


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ЛабМентор: виртуальный ассистент лабораторного практикума"
    )
    parser.add_argument("--host", default="127.0.0.1", help="адрес веб-сервера")
    parser.add_argument("--port", type=int, default=7860, help="порт веб-сервера")
    parser.add_argument("--share", action="store_true", help="создать публичную ссылку Gradio")
    args = parser.parse_args()

    settings = Settings.from_env()
    for problem in check_setup(settings):
        print(f"Внимание: {problem}")
    assistant = LabAssistant(settings)
    print(f"Провайдер LLM: {settings.provider}, модель: {settings.chat_model}")
    print(f"База знаний: {assistant.kb_status}")
    demo = build_ui(assistant)
    demo.queue(default_concurrency_limit=4).launch(
        server_name=args.host, server_port=args.port, share=args.share, **launch_options(settings)
    )


if __name__ == "__main__":
    main()
