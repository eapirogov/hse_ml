"""Вспомогательные функции для задания 1: оформление графиков и методы импутации.

Вынесено из ноутбука отдельным модулем, чтобы ячейки оставались читаемыми,
а функции можно было протестировать без запуска Jupyter.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import lagrange

# --- Русские названия месяцев (locale на macOS ненадёжен, задаём вручную) ---
MONTHS_SHORT = ["янв", "фев", "мар", "апр", "май", "июн",
                "июл", "авг", "сен", "окт", "ноя", "дек"]
MONTHS_FULL = ["январь", "февраль", "март", "апрель", "май", "июнь",
               "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]

# Румбы для розы ветров
RUMBS_8 = ["С", "СВ", "В", "ЮВ", "Ю", "ЮЗ", "З", "СЗ"]
RUMBS_16 = ["С", "ССВ", "СВ", "ВСВ", "В", "ВЮВ", "ЮВ", "ЮЮВ",
            "Ю", "ЮЮЗ", "ЮЗ", "ЗЮЗ", "З", "ЗСЗ", "СЗ", "ССЗ"]


def style_axes(ax, xlabel: str, ylabel: str, title: str | None = None,
               grid_axis: str = "both", arrows: bool = True) -> None:
    """Единое оформление осей по требованиям методички.

    Закрывает три явных требования задания:
      * подписи осей (на русском, с единицами измерения);
      * «клетки» — сетка major + minor;
      * «направление изменения» осей — стрелки на концах (аналог arrows() из R).
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Клетки: основная сетка сплошная, вспомогательная — пунктиром.
    ax.minorticks_on()
    if grid_axis in ("both", "x"):
        ax.grid(True, which="major", axis="x", ls="-", lw=0.7, alpha=0.55, color="0.55")
        ax.grid(True, which="minor", axis="x", ls=":", lw=0.5, alpha=0.45, color="0.7")
    if grid_axis in ("both", "y"):
        ax.grid(True, which="major", axis="y", ls="-", lw=0.7, alpha=0.55, color="0.55")
        ax.grid(True, which="minor", axis="y", ls=":", lw=0.5, alpha=0.45, color="0.7")
    ax.set_axisbelow(True)

    if arrows:
        # Убираем верхнюю и правую рамки, чтобы стрелки читались как оси координат.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        arrow_style = dict(arrowstyle="-|>", color="black", lw=1.1)
        ax.annotate("", xy=(1.025, 0), xycoords="axes fraction",
                    xytext=(0, 0), textcoords="axes fraction",
                    arrowprops=arrow_style, annotation_clip=False)
        ax.annotate("", xy=(0, 1.03), xycoords="axes fraction",
                    xytext=(0, 0), textcoords="axes fraction",
                    arrowprops=arrow_style, annotation_clip=False)


# --------------------------------------------------------------------------
#                      Методы заполнения пропусков
# --------------------------------------------------------------------------

def fill_prev3(values: np.ndarray, known_mask: np.ndarray) -> np.ndarray:
    """Заполнение средним трёх предыдущих ИЗВЕСТНЫХ значений ряда.

    Каузальный метод: идём слева направо и опираемся только на реально
    известные значения. Восстановленные оценки в список опор НЕ добавляются,
    иначе метод начинает питаться собственными результатами и ошибка
    накапливается.

    Краевой случай: если перед пропуском известных значений меньше трёх,
    усредняем столько, сколько есть.
    """
    out = values.astype(float).copy()
    known_values: list[float] = []
    for i in range(len(out)):
        if known_mask[i]:
            known_values.append(float(values[i]))
        else:
            if known_values:
                out[i] = float(np.mean(known_values[-3:]))
            else:
                out[i] = np.nan  # опор слева ещё нет
    return out


def lagrange_local(x_known: np.ndarray, y_known: np.ndarray,
                   x_query: np.ndarray, k: int = 4) -> np.ndarray:
    """Локальная интерполяция Лагранжа по k ближайшим узлам.

    Глобальный полином Лагранжа по всем узлам непригоден (эффект Рунге),
    поэтому для каждой точки строим полином невысокой степени по k
    ближайшим известным узлам.
    """
    x_known = np.asarray(x_known, dtype=float)
    y_known = np.asarray(y_known, dtype=float)
    result = np.empty(len(x_query), dtype=float)

    for n, q in enumerate(np.asarray(x_query, dtype=float)):
        idx = np.argsort(np.abs(x_known - q))[:k]
        idx = np.sort(idx)
        poly = lagrange(x_known[idx], y_known[idx])
        result[n] = float(np.polyval(poly, q))
    return result


def error_stats(truth: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    """Базовые характеристики отклонений восстановленных значений от реальных."""
    err = np.asarray(truth, dtype=float) - np.asarray(restored, dtype=float)
    finite = err[np.isfinite(err)]
    return {
        "Средняя абс. ошибка, °C": float(np.mean(np.abs(finite))),
        "Макс. абс. ошибка, °C": float(np.max(np.abs(finite))),
        "Смещение, °C": float(np.mean(finite)),
        "СКО ошибки, °C": float(np.std(finite, ddof=1)),
    }


def sigma_outliers(err: np.ndarray, n_sigma: float = 2.0):
    """Поиск недостоверных значений правилом n-сигма по распределению ошибок.

    Возвращает (маска выбросов, среднее, СКО, нижняя граница, верхняя граница).
    """
    err = np.asarray(err, dtype=float)
    mu = float(np.mean(err))
    sd = float(np.std(err, ddof=1))
    low, high = mu - n_sigma * sd, mu + n_sigma * sd
    mask = np.abs(err - mu) > n_sigma * sd
    return mask, mu, sd, low, high


def iqr_outliers(err: np.ndarray):
    """Поиск выбросов по правилу межквартильного размаха (как в box-plot).

    Возвращает (маска выбросов, Q1, Q3, IQR, нижний ус, верхний ус).
    """
    err = np.asarray(err, dtype=float)
    q1, q3 = np.percentile(err, [25, 75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (err < low) | (err > high)
    return mask, float(q1), float(q3), float(iqr), float(low), float(high)
