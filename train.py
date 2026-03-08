# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv"]
# ///
"""
Запускает LoRA-тренировку через ai-toolkit.

Читает .env, подставляет переменные в YAML-конфиг,
записывает разрешённый конфиг во временный файл и передаёт его ai-toolkit.

Режимы:
  --mode test   Быстрый тест (~5-10 мин): 50 шагов, rank 4, 512×1024
  --mode full   Полная тренировка (~1-2 ч): 1000 шагов, rank 16, 960×1952

Запуск:
    uv run train.py --mode test
    uv run train.py --mode full
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# UTF-8 для вывода в консоль (Windows cp1251)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
AI_TOOLKIT_DIR = ROOT / "vendor" / "ai-toolkit"
CONFIG_DIR = ROOT / "config"


def check_setup() -> None:
    if not AI_TOOLKIT_DIR.exists():
        print("[ОШИБКА] vendor/ai-toolkit не найден.")
        print("         Сначала запусти: uv run setup_training.py")
        sys.exit(1)

    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print("[ОШИБКА] .venv не найден — зависимости не установлены.")
        print("         Сначала запусти: uv run setup_training.py")
        sys.exit(1)


def load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        print("[ОШИБКА] Файл .env не найден.")
        print("         Создай его из шаблона: cp .env.example .env")
        sys.exit(1)
    load_dotenv(env_file)
    # Всегда добавляем PROJECT_ROOT чтобы YAML мог на него ссылаться
    os.environ["PROJECT_ROOT"] = str(ROOT).replace("\\", "/")


def resolve_config(template_path: Path) -> Path:
    """Подставляет ${VAR} из окружения в YAML и сохраняет во временный файл."""
    template = template_path.read_text(encoding="utf-8")

    def substitute(match: re.Match) -> str:
        var = match.group(1)
        value = os.environ.get(var)
        if value is None:
            print(f"[!] Переменная ${{{var}}} не задана в .env — оставляю как есть")
            return match.group(0)
        return value

    resolved = re.sub(r"\$\{(\w+)\}", substitute, template)

    # Пишем во временный файл рядом с конфигами (не в системный temp)
    out_path = CONFIG_DIR / f"_resolved_{template_path.name}"
    out_path.write_text(resolved, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="CP2077 Tarot LoRA training")
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        default="full",
        help="test = быстрый прогон (~5-10 мин) | full = полное обучение (~1-2 ч)",
    )
    args = parser.parse_args()

    check_setup()
    load_env()

    template = CONFIG_DIR / f"train_{args.mode}.yaml"
    if not template.exists():
        print(f"[ОШИБКА] Конфиг не найден: {template}")
        sys.exit(1)

    resolved = resolve_config(template)

    label = "ТЕСТОВЫЙ ПРОГОН" if args.mode == "test" else "ПОЛНАЯ ТРЕНИРОВКА"
    output_dir = ROOT / ("output_test" if args.mode == "test" else "output")

    print("=" * 60)
    print(f"  {label}")
    print(f"  Конфиг:    {resolved}")
    print(f"  Результат: {output_dir}")
    print("=" * 60 + "\n")

    result = subprocess.run(
        [
            str(ROOT / ".venv" / "Scripts" / "python.exe"),
            str(AI_TOOLKIT_DIR / "run.py"),
            str(resolved),
        ],
        cwd=AI_TOOLKIT_DIR,
        env={**os.environ, "VIRTUAL_ENV": str(ROOT / ".venv")},
    )

    # Удаляем временный конфиг
    resolved.unlink(missing_ok=True)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("  Тренировка завершена!")
        print(f"  LoRA-файлы:  {output_dir}")
        print(f"  Превью:      {output_dir / 'samples'}")
        if args.mode == "test":
            print("\n  Если всё без ошибок — запускай полную тренировку:")
            print("  uv run train.py --mode full")
        print("=" * 60)
    else:
        print(f"\n[ОШИБКА] Тренировка завершилась с кодом {result.returncode}")
        print("  Частые причины:")
        print("  - CUDA out of memory -> уменьши разрешение в .env")
        print("  - Неверный путь модели -> проверь HF_MODEL_PATH в .env")
        print("    Укажи repo id (black-forest-labs/FLUX.1-dev) или путь к snapshot после: hf download black-forest-labs/FLUX.1-dev")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
