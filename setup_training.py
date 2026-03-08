# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv"]
# ///
"""
Устанавливает окружение для дообучения LoRA через ai-toolkit.

Что делает:
  1. Клонирует ostris/ai-toolkit в vendor/ai-toolkit/
  2. Устанавливает ML-зависимости (torch CUDA + остальное) через uv sync --extra training
  3. Проверяет наличие файлов моделей из .env

Запуск:
    uv run setup_training.py
"""

import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).parent
VENDOR_DIR = ROOT / "vendor"
AI_TOOLKIT_DIR = VENDOR_DIR / "ai-toolkit"
ENV_FILE = ROOT / ".env"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    print(f"  > {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        print(f"\n[ОШИБКА] Команда завершилась с кодом {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def check_prerequisites() -> None:
    for tool in ["git", "uv"]:
        if run([tool, "--version"], check=False) != 0:
            urls = {"git": "https://git-scm.com/", "uv": "https://docs.astral.sh/uv/"}
            print(f"[ОШИБКА] {tool} не найден. Установи: {urls[tool]}")
            sys.exit(1)


def clone_or_update_ai_toolkit() -> None:
    VENDOR_DIR.mkdir(exist_ok=True)
    if AI_TOOLKIT_DIR.exists():
        print("\n[ai-toolkit] Уже клонирован — обновляем...")
        run(["git", "pull"], cwd=AI_TOOLKIT_DIR)
    else:
        print("\n[ai-toolkit] Клонируем репозиторий...")
        run(["git", "clone", "https://github.com/ostris/ai-toolkit.git", str(AI_TOOLKIT_DIR)])

    print("[ai-toolkit] Инициализируем сабмодули...")
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=AI_TOOLKIT_DIR)


def install_dependencies() -> None:
    print("\n[uv] Устанавливаем зависимости (torch CUDA + пакеты)...")
    print("     Первая установка может занять 10-20 минут (~2.5 GB для torch)\n")
    run(["uv", "sync", "--extra", "training"], cwd=ROOT)

    # Зависимости ai-toolkit (oyaml, cv2, albumentations и др.; torch не в списке — не перезапишется)
    print("\n[uv] Устанавливаем зависимости ai-toolkit...")
    run(
        ["uv", "pip", "install", "-r", str(AI_TOOLKIT_DIR / "requirements.txt")],
        cwd=ROOT,
    )


def create_output_dirs() -> None:
    for d in ["output", "output_test"]:
        (ROOT / d).mkdir(exist_ok=True)
    print("\n[dirs] Папки output/ и output_test/ созданы")


def verify_model_paths() -> None:
    if not ENV_FILE.exists():
        print("\n[!] Файл .env не найден — создай его из .env.example")
        return

    env = dotenv_values(ENV_FILE)
    val = env.get("HF_MODEL_PATH", "").strip()
    print("\n[models] Проверяем модель из .env (HF_MODEL_PATH):")
    if not val:
        print("  [!] HF_MODEL_PATH не задан — укажи repo id (black-forest-labs/FLUX.1-dev) или путь к snapshot")
        return

    # Repo id: org/repo — не проверяем локально
    if "/" in val and not Path(val).exists():
        print(f"  [OK] Repo: {val} (будет загружен при первом запуске)")
        return

    path = Path(val)
    if not path.exists():
        print(f"  [!] Путь не найден: {path}")
        print("      Скачай модель: hf download black-forest-labs/FLUX.1-dev")
        return

    if path.is_file():
        print(f"  [!] Указан файл — нужна директория (snapshot) или repo id: {path}")
        return

    print(f"  Путь: {path}")
    # Локальная папка — проверяем наличие подпапок FLUX
    for sub in ("transformer", "vae"):
        if (path / sub).exists():
            print(f"  [OK] {sub}/")
        else:
            print(f"  [!] Ожидается подпапка: {path / sub}")


def main() -> None:
    print("=" * 60)
    print("  Настройка окружения CP2077 Tarot LoRA")
    print("=" * 60)

    check_prerequisites()
    clone_or_update_ai_toolkit()
    install_dependencies()
    create_output_dirs()
    verify_model_paths()

    print("\n" + "=" * 60)
    print("  Готово! Запускай тренировку:")
    print()
    print("  Тест  (~5-10 мин):  uv run train.py --mode test")
    print("  Полная (~1-2 ч):   uv run train.py --mode full")
    print("=" * 60)


if __name__ == "__main__":
    main()
