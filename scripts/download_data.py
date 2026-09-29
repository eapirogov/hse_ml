"""Скачивание исходных данных для задания 1 по машинному обучению.

Источник: Open-Meteo Archive API (реанализ ERA5), https://open-meteo.com/
Лицензия данных: CC-BY 4.0, доступ без ключа.

Набор: почасовые метеорологические наблюдения по Перми за 2024 год.
Скрипт идемпотентный: если файл уже скачан, повторный запрос не выполняется.

Запуск:  .venv/bin/python scripts/download_data.py
"""

from pathlib import Path

import requests

# Пермь: координаты кампуса НИУ ВШЭ. API вернёт ближайший узел сетки ERA5 (58.03, 56.27).
LATITUDE = 58.01
LONGITUDE = 56.25
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

# wind_speed_unit=ms — сразу метры в секунду, чтобы не переводить из км/ч вручную.
URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&start_date={START_DATE}&end_date={END_DATE}"
    "&hourly=temperature_2m,relative_humidity_2m,surface_pressure,"
    "wind_speed_10m,wind_direction_10m,precipitation"
    "&timezone=Asia%2FYekaterinburg"
    "&wind_speed_unit=ms"
    "&format=csv"
)

TARGET = Path(__file__).resolve().parent.parent / "data" / "raw" / "perm_2024_hourly.csv"


def main() -> None:
    if TARGET.exists():
        print(f"Файл уже существует: {TARGET}")
        print(f"Размер: {TARGET.stat().st_size / 1024:.1f} КБ, "
              f"строк: {sum(1 for _ in TARGET.open())}")
        print("Повторное скачивание не требуется.")
        return

    print(f"Запрос к Open-Meteo Archive API...\n{URL}\n")
    response = requests.get(URL, timeout=120)
    response.raise_for_status()

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(response.text, encoding="utf-8")

    n_lines = sum(1 for _ in TARGET.open())
    print(f"Сохранено: {TARGET}")
    print(f"Размер: {TARGET.stat().st_size / 1024:.1f} КБ, строк: {n_lines}")
    print(f"Наблюдений (без 4 служебных строк шапки): {n_lines - 4}")


if __name__ == "__main__":
    main()
