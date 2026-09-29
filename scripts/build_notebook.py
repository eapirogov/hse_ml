"""Сборка ноутбука hw1_perm_weather.ipynb из списка ячеек.

Ноутбук собирается скриптом, а не верстается руками: так проще править
формулировки и переставлять разделы, не ломая JSON.

Запуск:  .venv/bin/python scripts/build_notebook.py
"""

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append((MD, text.strip("\n")))


def code(text: str) -> None:
    cells.append((CODE, text.strip("\n")))


# ===========================================================================
# Раздел 0. Титул
# ===========================================================================
md(r"""
# Машинное обучение. Задание 1

## Подготовка данных: загрузка и визуализация, импутирование и интерполяция

Выполнил: Егор Пирогов
НИУ ВШЭ — Пермь, кафедра информационных технологий в бизнесе, 2025

### Набор данных

Взят собственный набор вместо предложенных преподавателем (доп. часть, +1 балл).

Источник — [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api),
реанализ ERA5 (ECMWF), лицензия CC-BY 4.0. Точка — Пермь (58.03° с.ш., 56.27° в.д.),
период — весь 2024 год с часовым шагом, то есть 8784 наблюдения. Переменные:
температура, относительная влажность, приземное давление, скорость и направление
ветра, осадки.

Запрос к API вынесен в `scripts/download_data.py`, ноутбук читает локальный CSV
и не требует интернета.

Целевая оценка — 9 баллов: 8 за базовую часть и 1 за свой набор данных с
дополнительным типом диаграммы.
""")

md("### Импорт библиотек и настройка оформления")

code(r"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import lagrange, CubicSpline

# Вспомогательные функции работы: оформление осей и методы импутации.
sys.path.insert(0, str(Path.cwd().parent / "scripts"))
from hw_utils import (
    style_axes, fill_prev3, error_stats,
    sigma_outliers, iqr_outliers,
    MONTHS_SHORT, MONTHS_FULL, RUMBS_8, RUMBS_16,
)

DATA = Path.cwd().parent / "data" / "raw" / "perm_2024_hourly.csv"
FIG = Path.cwd().parent / "figures"
FIG.mkdir(exist_ok=True)

# Кириллица в подписях и корректный знак «минус» для отрицательных температур.
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.facecolor"] = "white"
sns.set_style("white")

np.random.seed(42)  # воспроизводимость
pd.set_option("display.precision", 3)

print("Проверка кириллицы: Пермь, температура −32.5 °C")
print("pandas", pd.__version__, "| numpy", np.__version__)
""")

# ===========================================================================
# Раздел 1. Данные
# ===========================================================================
md(r"""
---
## Пункт 1. Исходные данные

Файл скачивается один раз скриптом `scripts/download_data.py` и хранится в
репозитории работы, чтобы результат воспроизводился без обращения к сети.
""")

code(r"""
if not DATA.exists():
    raise FileNotFoundError(
        f"Не найден файл {DATA}.\n"
        "Выполните: .venv/bin/python scripts/download_data.py"
    )

print(f"Файл: {DATA.name}")
print(f"Размер: {DATA.stat().st_size / 1024:.1f} КБ")
print(f"Строк в файле: {sum(1 for _ in DATA.open())}")
print()
print("Первые строки файла как есть:")
print("".join(DATA.open(encoding="utf-8").readlines()[:5]))
""")

md(r"""
Структура CSV от Open-Meteo нестандартная: первые три строки служебные
(две строки метаданных о точке наблюдения и пустая строка-разделитель), и лишь
четвёртая строка содержит имена колонок. Поэтому при чтении указывается
`skiprows=3` — иначе pandas примет метаданные за заголовок и все столбцы
получат тип `object` вместо чисел.
""")

# ===========================================================================
# Раздел 2. Загрузка и преобразование типов
# ===========================================================================
md(r"""
---
## Пункт 2. Загрузка данных, преобразование типов, диаграммы

### 2.1. Паспорт точки наблюдения
""")

code(r"""
meta = pd.read_csv(DATA, nrows=1)
print("Метаданные точки наблюдения:")
for col in meta.columns:
    print(f"  {col:24s} = {meta[col].iloc[0]}")
""")

md("### 2.2. Чтение таблицы наблюдений и приведение типов")

code(r"""
df = pd.read_csv(DATA, skiprows=3)
print("Исходные имена колонок (с единицами измерения):")
for c in df.columns:
    print("   ", c)

# Короткие имена для кода + отдельный словарь единиц для подписей осей.
df.columns = ["time", "temp", "humidity", "pressure",
              "wind_speed", "wind_dir", "precip"]

UNITS = {
    "temp": "Температура воздуха, °C",
    "humidity": "Относительная влажность, %",
    "pressure": "Приземное давление, гПа",
    "wind_speed": "Скорость ветра, м/с",
    "wind_dir": "Направление ветра, °",
    "precip": "Осадки, мм",
}

# Ключевое преобразование типов: строка -> datetime -> индекс временного ряда.
print("\nТип колонки time до преобразования:", df["time"].dtype)
df["time"] = pd.to_datetime(df["time"])
print("Тип колонки time после преобразования:", df["time"].dtype)

df = df.set_index("time").sort_index()

# Производные признаки для группировок и диаграмм.
df["month"] = df.index.month
df["hour"] = df.index.hour
df["doy"] = df.index.dayofyear
df["date"] = df.index.date

df.head()
""")

md("### 2.3. Контроль качества данных")

code(r"""
print(f"Размерность таблицы: {df.shape[0]} наблюдений × {df.shape[1]} признаков")
print(f"Период: {df.index.min()} — {df.index.max()}")
print(f"366 суток × 24 часа = {366 * 24} (2024 — високосный год)")

print("\nПропуски по столбцам:")
print(df[list(UNITS)].isna().sum().to_string())

full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
missing_stamps = full_range.difference(df.index)
print(f"\nПропущенных отметок времени в сетке: {len(missing_stamps)}")
print(f"Ряд регулярный и непрерывный: {missing_stamps.empty}")

print("\nОписательные статистики:")
df[list(UNITS)].describe().T
""")

md(r"""
Следствие для пункта 4. В наборе данных нет ни одного пропуска:
ERA5 — это реанализ, то есть результат работы модели, усваивающей наблюдения,
и он по построению даёт полное поле без лакун. Поэтому пропуски в пункте 4
создаются искусственно, прореживанием заведомо известного участка ряда.
Удалённые значения при этом известны точно, и качество каждого метода восстановления можно измерить, сравнив
оценку с настоящим значением.
""")

# ===========================================================================
# Раздел 3. Обязательные диаграммы
# ===========================================================================
md(r"""
---
### 2.4. Обязательные виды диаграмм

На каждой диаграмме выполнены требования методички: использован весь годовой
ряд из 8784 наблюдений вместо минимальной тысячи (число указано в заголовке),
оси подписаны по-русски с единицами измерения, на концах осей нарисованы стрелки
направления (аналог `arrows()` из примеров на R), построена основная и
вспомогательная сетка. Оформление вынесено в функцию `style_axes()` модуля
`scripts/hw_utils.py`.

#### 2.4.1. Линейная диаграмма — годовой ход температуры
""")

code(r"""
fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(df.index, df["temp"], lw=0.35, color="tab:blue",
        alpha=0.75, label="Почасовые значения")
ax.plot(df.index, df["temp"].rolling(24 * 7, center=True).mean(),
        lw=2.2, color="darkred", label="Скользящее среднее за 7 суток")
ax.axhline(0, color="grey", ls="--", lw=1, label="0 °C")

style_axes(ax, "Месяц 2024 года", UNITS["temp"],
           f"Годовой ход температуры воздуха, Пермь, 2024 (n = {len(df)} наблюдений)")

ax.set_xticks([pd.Timestamp(2024, m, 1) for m in range(1, 13)])
ax.set_xticklabels(MONTHS_SHORT)
ax.legend(loc="upper left", framealpha=0.9)

fig.savefig(FIG / "01_line_temperature.png")
plt.show()
""")

md(r"""
Ряд ведёт себя так, как и должен вести себя годовой ход температуры в умеренных
широтах: холодная половина года в начале и конце, тёплая — в середине. Тонкая
линия почасовых значений образует «облако» вокруг сглаженной кривой, и ширина
этого облака зимой заметно больше, чем летом.

#### 2.4.2. Столбчатая диаграмма — месячные суммы осадков и средняя температура
""")

code(r"""
monthly_precip = df["precip"].resample("ME").sum()
monthly_temp = df["temp"].resample("ME").mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

bars = axes[0].bar(range(1, 13), monthly_precip.values,
                   color="tab:blue", edgecolor="black", lw=0.6)
axes[0].bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
style_axes(axes[0], "Месяц", "Сумма осадков, мм",
           f"Месячные суммы осадков (по n = {len(df)} часовым значениям)",
           grid_axis="y")
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(MONTHS_SHORT)

colors = np.where(monthly_temp.values < 0, "tab:blue", "tab:red")
bars2 = axes[1].bar(range(1, 13), monthly_temp.values,
                    color=colors, edgecolor="black", lw=0.6)
axes[1].bar_label(bars2, fmt="%.1f", fontsize=8, padding=2)
axes[1].axhline(0, color="black", lw=1)
style_axes(axes[1], "Месяц", UNITS["temp"],
           "Средняя месячная температура", grid_axis="y")
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(MONTHS_SHORT)

fig.tight_layout()
fig.savefig(FIG / "02_bar_monthly.png")
plt.show()

print("Годовая сумма осадков: %.0f мм" % monthly_precip.sum())
""")

md("#### 2.4.3. Круговая диаграмма — повторяемость направлений ветра по 8 румбам")

code(r"""
# Сектор «север» охватывает 337.5°..22.5°, поэтому шкала сдвигается на полсектора.
shifted = (df["wind_dir"] + 22.5) % 360
sector = pd.cut(shifted, bins=np.arange(0, 361, 45),
                labels=RUMBS_8, right=False, include_lowest=True)
counts = sector.value_counts().reindex(RUMBS_8)

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(counts.values, labels=RUMBS_8, autopct="%1.1f%%",
       startangle=90, counterclock=False,
       colors=plt.cm.tab20(np.linspace(0, 1, 8)),
       wedgeprops=dict(edgecolor="white", lw=1.4),
       textprops=dict(fontsize=11))
ax.set_title(f"Повторяемость направлений ветра по румбам,\nПермь, 2024 "
             f"(n = {len(df)} наблюдений)")
ax.axis("equal")

fig.savefig(FIG / "03_pie_wind_sectors.png")
plt.show()

print("Повторяемость по румбам, %:")
print((counts / counts.sum() * 100).round(1).to_string())
""")

md(r"""
Для круговой диаграммы подписи осей, сетка и стрелки направления неприменимы:
у неё нет декартовых осей, а доли секторов подписаны непосредственно на
диаграмме в процентах. Требование методички об оформлении осей относится к
линейным и столбчатым диаграммам.

#### 2.4.4. Скрипичная диаграмма — распределение температуры по месяцам
""")

code(r"""
fig, axes = plt.subplots(2, 1, figsize=(13, 9))

sns.violinplot(data=df, x="month", y="temp", inner="box",
               cut=0, density_norm="width", hue="month",
               palette="coolwarm", legend=False, ax=axes[0])
style_axes(axes[0], "Месяц", UNITS["temp"],
           f"Распределение температуры по месяцам (n = {len(df)} наблюдений)",
           grid_axis="y")
axes[0].set_xticks(range(12))
axes[0].set_xticklabels(MONTHS_SHORT)
axes[0].axhline(0, color="grey", ls="--", lw=1)

season = {12: "Зима", 1: "Зима", 2: "Зима", 3: "Весна", 4: "Весна", 5: "Весна",
          6: "Лето", 7: "Лето", 8: "Лето", 9: "Осень", 10: "Осень", 11: "Осень"}
df["season"] = df["month"].map(season)
order = ["Зима", "Весна", "Лето", "Осень"]

sns.violinplot(data=df, x="season", y="humidity", order=order, inner="box",
               cut=0, density_norm="width", hue="season", palette="Blues",
               legend=False, ax=axes[1])
style_axes(axes[1], "Сезон", UNITS["humidity"],
           "Распределение относительной влажности по сезонам", grid_axis="y")

fig.tight_layout()
fig.savefig(FIG / "04_violin_temp_humidity.png")
plt.show()
""")

md(r"""
Скрипичная диаграмма показывает не только медиану и квартили (как box-plot),
но и форму распределения. Видно, что зимние месяцы дают широкие и
часто двугорбые «скрипки», а летние — узкие и компактные. Содержательная
интерпретация этого факта — в следующем разделе.
""")

# ===========================================================================
# Раздел 4. Качественный анализ
# ===========================================================================
md(r"""
---
## Пункт 3. Качественный анализ предметной области

Предметная область — приземная метеорология умеренных широт, точка наблюдения
в Предуралье. Каждый вывод ниже подкреплён числом, рассчитанным по данным.

### 3.1. Суточный ход температуры
""")

code(r"""
hourly = df.groupby("hour")["temp"].mean()
summer_h = df[df["month"].isin([6, 7, 8])].groupby("hour")["temp"].mean()
winter_h = df[df["month"].isin([12, 1, 2])].groupby("hour")["temp"].mean()

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(hourly.index, hourly.values, lw=2.4, color="black", marker="o",
        ms=4, label="Весь год")
ax.plot(summer_h.index, summer_h.values, lw=1.8, color="tab:red",
        marker="^", ms=4, label="Лето (VI–VIII)")
ax.plot(winter_h.index, winter_h.values, lw=1.8, color="tab:blue",
        marker="v", ms=4, label="Зима (XII–II)")

style_axes(ax, "Час суток (местное время, UTC+5)", UNITS["temp"],
           f"Средний суточный ход температуры (n = {len(df)} наблюдений)")
ax.set_xticks(range(0, 24, 2))
ax.legend()

fig.savefig(FIG / "05_daily_cycle.png")
plt.show()

print(f"Минимум среднего суточного хода: {hourly.idxmin()} ч ({hourly.min():.1f} °C)")
print(f"Максимум среднего суточного хода: {hourly.idxmax()} ч ({hourly.max():.1f} °C)")
print(f"Амплитуда суточного хода летом: {summer_h.max() - summer_h.min():.1f} °C")
print(f"Амплитуда суточного хода зимой: {winter_h.max() - winter_h.min():.1f} °C")
""")

md("### 3.2. Связи между метеорологическими величинами")

code(r"""
corr = df[list(UNITS)[:-1]].corr()

fig, ax = plt.subplots(figsize=(7.2, 5.6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, linewidths=0.6, linecolor="white",
            cbar_kws={"label": "Коэффициент корреляции Пирсона"}, ax=ax)
ax.set_title("Корреляционная матрица метеовеличин")
labels = ["Температура", "Влажность", "Давление", "Скорость ветра", "Направление"]
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_yticklabels(labels, rotation=0)

fig.savefig(FIG / "06_corr_matrix.png")
plt.show()

print(f"Корреляция температура ↔ влажность (весь ряд): {corr.loc['temp', 'humidity']:.3f}")
print(f"Корреляция осадки ↔ изменение давления: "
      f"{df['precip'].corr(df['pressure'].diff()):.3f}")

monthly_stats = df.groupby("month")[["temp", "humidity", "pressure"]].mean()
monthly_stats.index = MONTHS_FULL
print("\nСредние по месяцам:")
monthly_stats.round(1)
""")

md(r"""
### 3.3. Выводы по предметной области

1\. Климат резко континентальный: средняя температура +3.4 °C при размахе от
−32.5 до +30.9 °C, то есть годовая амплитуда больше 63 градусов. Пермь удалена
от океана, а Уральские горы ослабляют западный перенос.

2\. Суточный минимум приходится на 6 часов, максимум на 15 — не на полдень,
потому что почва отдаёт тепло и после максимума солнечной радиации. Летом
амплитуда суточного хода 7.2 °C, зимой 2.3 °C: зимой солнце низкое и часта
облачность. Поэтому для интерполяции в пункте 4 взят летний участок — на нём
разница между методами заметна.

3\. Температура и влажность связаны обратно: коэффициент корреляции
отрицательный. Тёплый воздух способен удержать больше водяного пара, поэтому
дневной нагрев снижает относительную влажность, а ночное остывание поднимает её
почти до 100 % — отсюда роса и туманы под утро.

4\. Осадки идут при падении давления, то есть при приближении циклона.

5\. Распределение осадков асимметрично: большинство часов нулевые, редкие дают
крупные значения. Среднее здесь смещено редкими ливнями, поэтому на столбчатой
диаграмме осадки просуммированы за месяц.

6\. Зимние «скрипки» шире летних и часто двугорбые: зимой Пермь попеременно под
арктическими вторжениями (морозы до −30 °C) и атлантическими циклонами
(оттепели около 0 °C). Летом режим один, поэтому распределение компактное.
""")

# ===========================================================================
# Раздел 5. Пропуски
# ===========================================================================
md(r"""
---
## Пункт 4. Методы заполнения пропущенных значений

### 4.1. Формирование пропусков прореживанием

Берётся непрерывный участок ряда температуры длиной 48 часов (двое летних
суток), из которого удаляется каждое второе значение. Получается
24 пропуска, что удовлетворяет требованию «не менее 20 значений».

Участок выбран летним осознанно: как показано в пункте 3, летом суточный ход
выражен максимально, поэтому восстанавливаемый сигнал гладкий и периодичный —
на нём различия между методами видны отчётливо.
""")

code(r"""
SEG_START, SEG_END = "2024-07-10", "2024-07-11"
seg = df.loc[SEG_START:SEG_END, "temp"]

y_full = seg.to_numpy(dtype=float)
x = np.arange(len(y_full), dtype=float)      # часы от начала участка

known_mask = np.zeros(len(y_full), dtype=bool)
known_mask[::2] = True                        # оставляем каждое второе
missing_mask = ~known_mask

x_known, y_known = x[known_mask], y_full[known_mask]
x_missing, y_truth = x[missing_mask], y_full[missing_mask]

work = np.where(known_mask, y_full, np.nan)   # ряд с пропусками

print(f"Участок: {seg.index[0]} — {seg.index[-1]}")
print(f"Длина участка:            {len(y_full)} значений")
print(f"Известные (каждое второе): {known_mask.sum()}")
print(f"Удалённые (пропуски):      {missing_mask.sum()}  ← требуется ≥ 20")
print(f"Температура на участке: от {y_full.min():.1f} до {y_full.max():.1f} °C")
""")

code(r"""
fig, ax = plt.subplots(figsize=(13, 4.8))

ax.plot(x, y_full, lw=1.1, color="grey", alpha=0.65, label="Исходный ряд")
ax.plot(x_known, y_known, "o", ms=7, color="tab:blue",
        label=f"Известные значения ({known_mask.sum()})")
ax.plot(x_missing, y_truth, "X", ms=8, color="tab:red",
        label=f"Удалённые значения ({missing_mask.sum()})")

style_axes(ax, "Часы от начала участка", UNITS["temp"],
           f"Схема прореживания: {SEG_START} — {SEG_END}, удалено каждое второе значение")
ax.legend()

fig.savefig(FIG / "07_gaps_scheme.png")
plt.show()
""")

md(r"""
### 4.2. Заполнение средним и медианой ряда

Простейший подход: все пропуски заменяются одной константой — средним или
медианой известных значений участка.
""")

code(r"""
restored = {}   # сюда собираются результаты всех методов

restored["Среднее"] = np.full(len(x_missing), y_known.mean())
restored["Медиана"] = np.full(len(x_missing), np.median(y_known))

print(f"Среднее известных значений: {y_known.mean():.2f} °C")
print(f"Медиана известных значений: {np.median(y_known):.2f} °C")

fig, ax = plt.subplots(figsize=(13, 4.8))
ax.plot(x, y_full, lw=1.1, color="grey", alpha=0.6, label="Реальный ряд")
ax.plot(x_known, y_known, "o", ms=6, color="tab:blue", label="Известные значения")
ax.plot(x_missing, y_truth, "X", ms=8, color="tab:red", label="Реальные (удалённые)")
ax.plot(x_missing, restored["Среднее"], "s", ms=6, color="tab:green",
        label="Восстановлено средним")
ax.plot(x_missing, restored["Медиана"], "d", ms=6, color="tab:purple",
        label="Восстановлено медианой")

style_axes(ax, "Часы от начала участка", UNITS["temp"],
           "Заполнение средним и медианой ряда")
ax.legend(ncol=2, fontsize=9)

fig.savefig(FIG / "08_impute_mean_median.png")
plt.show()
""")

md(r"""
Восстановленные значения выстраиваются в горизонтальную «полку», тогда как
реальные образуют суточную волну. Метод сохраняет только общий уровень, теряя и
суточный ход, и локальный тренд; ошибка выходит порядка половины амплитуды.

### 4.3. Заполнение средним трёх предыдущих известных значений

Каузальный метод: для каждого пропуска берётся среднее трёх предыдущих
известных значений. Восстановленные оценки в число опорных не включаются —
иначе метод начал бы опираться на собственные результаты и ошибка накапливалась
бы вдоль ряда.
""")

code(r"""
prev3_full = fill_prev3(work, known_mask)
restored["3 предыдущих"] = prev3_full[missing_mask]

fig, ax = plt.subplots(figsize=(13, 4.8))
ax.plot(x, y_full, lw=1.1, color="grey", alpha=0.6, label="Реальный ряд")
ax.plot(x_known, y_known, "o", ms=6, color="tab:blue", label="Известные значения")
ax.plot(x_missing, y_truth, "X", ms=8, color="tab:red", label="Реальные (удалённые)")
ax.plot(x_missing, restored["3 предыдущих"], "s", ms=6, color="tab:orange",
        label="Среднее 3 предыдущих известных")

style_axes(ax, "Часы от начала участка", UNITS["temp"],
           "Заполнение средним трёх предыдущих известных значений")
ax.legend(ncol=2, fontsize=9)

fig.savefig(FIG / "09_impute_prev3.png")
plt.show()

diff = restored["3 предыдущих"] - y_truth
rising = diff[np.gradient(y_full)[missing_mask] > 0].mean()
falling = diff[np.gradient(y_full)[missing_mask] < 0].mean()
print(f"Среднее отклонение на участках роста температуры:   {rising:+.2f} °C")
print(f"Среднее отклонение на участках падения температуры: {falling:+.2f} °C")
""")

md(r"""
Метод учитывает локальный уровень ряда и потому заметно точнее константы, но
обладает систематическим запаздыванием: он смотрит только назад, поэтому на
участках роста температуры занижает значения, а на участках падения — завышает.
Числа под графиком показывают этот сдвиг явно: отклонения на растущих и
падающих участках имеют противоположные знаки. Это классическое свойство любого
каузального сглаживания.

### 4.4. Интерполяционный многочлен Лагранжа

Многочлен Лагранжа строится по всем 24 известным узлам сразу, то есть его
степень равна 23.
""")

code(r"""
with warnings.catch_warnings():
    warnings.simplefilter("ignore")          # переполнение здесь ожидаемо
    with np.errstate(over="ignore", invalid="ignore"):
        poly_global = lagrange(x_known, y_known)
        lagrange_global = np.polyval(poly_global, x_missing)

restored["Лагранж (глобальный)"] = lagrange_global

print(f"Степень многочлена: {len(x_known) - 1}")
print(f"Реальная температура на участке: от {y_full.min():.1f} до {y_full.max():.1f} °C")
print(f"Максимум |значения| глобального многочлена: {np.max(np.abs(lagrange_global)):.3e} °C")
print(f"Максимальная ошибка: {np.max(np.abs(lagrange_global - y_truth)):.3e} °C")
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

axes[0].plot(x, y_full, lw=1.2, color="grey", label="Реальный ряд")
axes[0].plot(x_known, y_known, "o", ms=6, color="tab:blue", label="Известные узлы")
axes[0].plot(x_missing, lagrange_global, "v", ms=7, color="tab:red",
             label="Глобальный Лагранж")
axes[0].set_ylim(y_full.min() - 3, y_full.max() + 3)
style_axes(axes[0], "Часы от начала участка", UNITS["temp"],
           "В масштабе данных: оценки уходят за пределы графика")
axes[0].legend(fontsize=9)

axes[1].semilogy(x_missing, np.abs(lagrange_global), "v-", ms=6,
                 color="tab:red", label="|Глобальный Лагранж|")
axes[1].axhline(y_full.max(), color="green", ls="--",
                label=f"Реальный максимум ({y_full.max():.1f} °C)")
style_axes(axes[1], "Часы от начала участка", "Модуль значения, °C (лог. шкала)",
           "Истинный масштаб оценок: эффект Рунге")
axes[1].legend(fontsize=9)

fig.tight_layout()
fig.savefig(FIG / "10_lagrange_global_runge.png")
plt.show()
""")

md(r"""
Результат непригоден: при реальных температурах в диапазоне 17–28 °C
многочлен выдаёт значения порядка 10⁴–10⁵ градусов, причём ошибка нарастает
к краям участка. Это эффект Рунге — осцилляции интерполяционного многочлена
высокой степени между узлами, особенно сильные вблизи концов отрезка, — вместе
с численной неустойчивостью вычисления коэффициентов многочлена 23-й степени.
Тот же эффект показан в материалах курса на примере ряда заявок по ипотечным
кредитам.

Отсюда вывод: интерполяционный многочлен высокой степени для заполнения
пропусков непригоден. Нужен метод, у которого степень не растёт вместе с
числом узлов, — таким является кубический сплайн.

### 4.5. Интерполяция кубическими сплайнами
""")

code(r"""
cs = CubicSpline(x_known, y_known)            # bc_type="not-a-knot" по умолчанию
restored["Кубический сплайн"] = cs(x_missing)

x_dense = np.linspace(x_known[0], x_known[-1], 600)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(x, y_full, lw=1.2, color="grey", alpha=0.7, label="Реальный ряд")
ax.plot(x_dense, cs(x_dense), lw=1.6, color="tab:green",
        label="Кубический сплайн (непрерывная кривая)")
ax.plot(x_known, y_known, "o", ms=6, color="tab:blue", label="Известные узлы")
ax.plot(x_missing, y_truth, "X", ms=9, color="tab:red", label="Реальные (удалённые)")
ax.plot(x_missing, restored["Кубический сплайн"], "s", ms=6, color="darkgreen",
        label="Восстановлено сплайном")

style_axes(ax, "Часы от начала участка", UNITS["temp"],
           "Интерполяция кубическими сплайнами")
ax.legend(ncol=2, fontsize=9)

fig.savefig(FIG / "12_impute_spline.png")
plt.show()

s = error_stats(y_truth, restored["Кубический сплайн"])
print(f"Средняя абсолютная ошибка сплайна: {s['Средняя абс. ошибка, °C']:.3f} °C")

# Тот же результат средствами pandas — одной строкой.
pandas_spline = (pd.Series(work, index=seg.index)
                 .interpolate(method="spline", order=3))
print(f"Проверка через pandas .interpolate(method='spline'): "
       f"средняя ошибка {np.mean(np.abs(pandas_spline.to_numpy()[missing_mask] - y_truth)):.3f} °C")
""")

md(r"""
Сплайн — кусочно-кубическая функция: на каждом промежутке между узлами строится
свой многочлен 3-й степени, а коэффициенты подбираются из условий гладкой
стыковки. Благодаря непрерывности второй производной кривая получается гладкой,
но, в отличие от глобального многочлена Лагранжа, степень нигде не растёт с
числом узлов — именно поэтому сплайн устойчив там, где Лагранж расходится.

### 4.6. Сводное сравнение всех методов
""")

code(r"""
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(x, y_full, lw=1.4, color="grey", alpha=0.55, label="Реальный ряд", zorder=1)
ax.plot(x_known, y_known, "o", ms=7, color="black", label="Известные узлы", zorder=5)
ax.plot(x_missing, y_truth, "X", ms=11, color="tab:red",
        label="Реальные (удалённые)", zorder=6)

styles = {
    "Среднее": ("s", "tab:green"),
    "Медиана": ("d", "tab:purple"),
    "3 предыдущих": ("^", "tab:orange"),
    "Кубический сплайн": ("P", "darkgreen"),
}
for name, (marker, color) in styles.items():
    ax.plot(x_missing, restored[name], marker, ms=6, color=color,
            alpha=0.9, label=name, zorder=4)

style_axes(ax, "Часы от начала участка", UNITS["temp"],
           "Сравнение методов восстановления пропущенных значений")
# Легенда вынесена под график, чтобы не перекрывать точки данных.
ax.legend(ncol=4, fontsize=9, loc="upper center",
          bbox_to_anchor=(0.5, -0.16), frameon=False)

fig.savefig(FIG / "13_all_methods.png")
plt.show()
""")

code(r"""
order = ["Среднее", "Медиана", "3 предыдущих",
         "Лагранж (глобальный)", "Кубический сплайн"]

table = pd.DataFrame(
    {name: error_stats(y_truth, restored[name]) for name in order}
).T
table = table.sort_values("Средняя абс. ошибка, °C")

print("Отклонения восстановленных значений от реальных (24 значения):\n")
table
""")

md(r"""
Методы выстраиваются в устойчивый порядок качества:

1. Кубический сплайн — ошибка в доли градуса. Он учитывает форму кривой и
   опирается на узлы по обе стороны от пропуска.
2. **Среднее трёх предыдущих** — ошибка порядка полутора градусов: локальный
   уровень схвачен, но есть систематическое запаздывание.
3. **Медиана и среднее** — ошибка порядка 2.5–3 °C, сопоставимая с половиной
   амплитуды суточного хода: структура ряда потеряна полностью.
4. **Глобальный Лагранж** — ошибка астрономическая, метод неприменим.

Причина такого порядка в том, что температура — гладкий процесс с суточной
периодичностью, поэтому методы, восстанавливающие форму кривой, на нём и
выигрывают.

На другой ряд этот вывод не переносится. Если бы восстанавливались осадки,
порядок был бы иным: ряд разрывный, большинство значений нулевые, и сплайн дал
бы осцилляции с отрицательными значениями, физически невозможными. Метод
импутации нужно выбирать под свойства конкретной величины.
""")

# ===========================================================================
# Раздел 6. Недостоверные значения
# ===========================================================================
md(r"""
---
## Пункт 5. Поиск недостоверных значений

Как требует задание, разница между реальными и восстановленными значениями
анализируется для одного метода. Выбран кубический сплайн — один из
двух лучших по точности и наиболее универсальный.

### 5.1. Правило двух и трёх сигм
""")

code(r"""
method = "Кубический сплайн"
err = y_truth - restored[method]

mask2, mu, sd, low2, high2 = sigma_outliers(err, 2.0)
mask3, _, _, low3, high3 = sigma_outliers(err, 3.0)

print(f"Метод: {method}")
print(f"Число анализируемых отклонений: {len(err)}\n")
print(f"Среднее отклонение (смещение) μ = {mu:+.3f} °C")
print(f"Стандартное отклонение       σ = {sd:.3f} °C\n")
print(f"Границы 2σ: [{low2:+.3f}; {high2:+.3f}] °C → выходят за пределы: {mask2.sum()}")
print(f"Границы 3σ: [{low3:+.3f}; {high3:+.3f}] °C → выходят за пределы: {mask3.sum()}")

if mask2.any():
    print("\nНедостоверные значения по правилу 2σ:")
    for i in np.where(mask2)[0]:
        ts = seg.index[missing_mask][i]
        print(f"  {ts:%d.%m.%Y %H:%M}  реальное {y_truth[i]:6.2f} °C,  "
              f"восстановлено {restored[method][i]:6.2f} °C,  отклонение {err[i]:+.2f} °C")
""")

code(r"""
times = seg.index[missing_mask]

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(times, err, "o-", ms=6, lw=1.2, color="tab:blue", label="Отклонение")

ax.axhline(mu, color="black", lw=1.4, label=f"Среднее μ = {mu:+.2f}")
for lvl, ls, c, lbl in [(high2, "--", "tab:orange", "μ ± 2σ"),
                        (low2, "--", "tab:orange", None),
                        (high3, ":", "tab:red", "μ ± 3σ"),
                        (low3, ":", "tab:red", None)]:
    ax.axhline(lvl, color=c, ls=ls, lw=1.4, label=lbl)

ax.fill_between(times, low2, high2, color="tab:green", alpha=0.08)

if mask2.any():
    ax.plot(times[mask2], err[mask2], "o", ms=13, mfc="none",
            mec="red", mew=2.2, label="Недостоверные (вне 2σ)")
    for i in np.where(mask2)[0]:
        # Подпись сдвигается влево, если точка близка к правому краю графика.
        to_right = i < len(times) - 3
        ax.annotate(f"{times[i]:%d.%m %H:%M}\n{err[i]:+.2f} °C",
                    xy=(times[i], err[i]),
                    xytext=(10, 14) if to_right else (-14, 16),
                    ha="left" if to_right else "right",
                    textcoords="offset points", fontsize=8.5, color="darkred")

style_axes(ax, "Время", "Отклонение (реальное − восстановленное), °C",
           f"Правило 2σ/3σ для отклонений: {method}")
ax.legend(ncol=2, fontsize=9)
fig.autofmt_xdate()

fig.savefig(FIG / "14_sigma_rule.png")
plt.show()
""")

md("### 5.2. Box-plot и скрипичная диаграмма отклонений")

code(r"""
mask_iqr, q1, q3, iqr, low_iqr, high_iqr = iqr_outliers(err)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

bp = axes[0].boxplot(err, orientation="vertical", patch_artist=True, widths=0.45,
                     flierprops=dict(marker="o", markerfacecolor="red",
                                     markersize=9, markeredgecolor="darkred"))
bp["boxes"][0].set_facecolor("lightsteelblue")
style_axes(axes[0], "Кубический сплайн",
           "Отклонение, °C", "Box-plot отклонений", grid_axis="y")
axes[0].set_xticks([1]); axes[0].set_xticklabels(["Отклонения"])

sns.violinplot(y=err, inner="box", cut=0, color="lightsteelblue", ax=axes[1])
axes[1].plot(np.zeros(mask_iqr.sum()), err[mask_iqr], "o", ms=9,
             color="red", mec="darkred", label="Выбросы по IQR")
style_axes(axes[1], "Кубический сплайн", "Отклонение, °C",
           "Скрипичная диаграмма отклонений", grid_axis="y")
if mask_iqr.any():
    axes[1].legend(fontsize=9)

fig.tight_layout()
fig.savefig(FIG / "15_boxplot_violin_errors.png")
plt.show()

print(f"Q1 = {q1:+.3f} °C,  Q3 = {q3:+.3f} °C,  IQR = {iqr:.3f} °C")
print(f"Границы усов: [{low_iqr:+.3f}; {high_iqr:+.3f}] °C")
print(f"\nВыбросов по правилу 1.5·IQR: {mask_iqr.sum()}")
print(f"Недостоверных по правилу 2σ:  {mask2.sum()}")
print(f"Недостоверных по правилу 3σ:  {mask3.sum()}")
""")

md("### 5.3. Сравнение с заведомо неточным методом")

code(r"""
rows = []
for name in ["Кубический сплайн", "3 предыдущих", "Среднее"]:
    e = y_truth - restored[name]
    m2, mu_, sd_, _, _ = sigma_outliers(e, 2.0)
    m3, *_ = sigma_outliers(e, 3.0)
    mi, *_ = iqr_outliers(e)
    rows.append({
        "Метод": name,
        "μ, °C": round(float(np.mean(e)), 3),
        "σ, °C": round(float(np.std(e, ddof=1)), 3),
        "Вне 2σ": int(m2.sum()),
        "Вне 3σ": int(m3.sum()),
        "Выбросов по IQR": int(mi.sum()),
    })

comparison = pd.DataFrame(rows).set_index("Метод")
print("Сравнение критериев недостоверности для разных методов:\n")
comparison
""")

code(r"""
# Проверка сигма-правила на полном годовом ряде (8784 значения).
t = df["temp"].to_numpy()
t_mu, t_sd = t.mean(), t.std(ddof=1)
out2 = np.abs(t - t_mu) > 2 * t_sd
out3 = np.abs(t - t_mu) > 3 * t_sd

print(f"Годовой ряд температуры: μ = {t_mu:.2f} °C, σ = {t_sd:.2f} °C")
print(f"Вне 2σ: {out2.sum()} значений ({out2.mean() * 100:.1f} %)")
print(f"Вне 3σ: {out3.sum()} значений ({out3.mean() * 100:.1f} %)")
print("\nДля нормального распределения ожидалось бы ≈ 4.6 % вне 2σ и ≈ 0.3 % вне 3σ.")

fig, ax = plt.subplots(figsize=(12, 4.6))
ax.hist(t, bins=80, color="lightsteelblue", edgecolor="black", lw=0.4)
for lvl, c, lbl in [(t_mu, "black", "μ"),
                    (t_mu - 2 * t_sd, "tab:orange", "μ ± 2σ"),
                    (t_mu + 2 * t_sd, "tab:orange", None),
                    (t_mu - 3 * t_sd, "tab:red", "μ ± 3σ"),
                    (t_mu + 3 * t_sd, "tab:red", None)]:
    ax.axvline(lvl, color=c, ls="--", lw=1.6, label=lbl)

style_axes(ax, UNITS["temp"], "Число наблюдений",
           f"Распределение температуры за год и границы сигм (n = {len(t)})")
ax.legend()

fig.savefig(FIG / "16_year_sigma_hist.png")
plt.show()
""")

md(r"""
### 5.4. Выводы по поиску недостоверных значений

Смещение μ близко к нулю: метод не ошибается систематически в одну сторону,
работает одинаково на подъёмах и спадах температуры. Правило 2σ выделяет
единичное отклонение, правило 3σ — ни одного. Критерий межквартильного размаха
даёт тот же результат, хотя устроен иначе: сигма-правило опирается на среднее и
стандартное отклонение и потому само чувствительно к выбросам, а квартили к ним
устойчивы. При малой выборке и ненормальном распределении IQR надёжнее.

Отсутствие точек вне 3σ закономерно: отклонений всего 24, и даже для строго
нормального распределения ожидаемое число таких точек меньше единицы.

Сравнение методов показывает ограничение критерия. У сплайна σ мала, границы
недостоверности узкие. У заполнения средним σ на порядок больше, а ошибки не
случайные, а структурные — они велики в моменты суточного максимума и минимума.
Но сигма-правило ищет отклонения, нетипичные на фоне общего разброса, поэтому у
равномерно плохого метода формальных выбросов может не оказаться вовсе. Критерий
недостоверности не заменяет оценку качества метода, а дополняет её.

На годовом ряде вне 2σ оказалось 324 значения (3.7 %) против ожидаемых для
нормального закона 4.6 %, а вне 3σ — ни одного против ожидаемых 0.3 %.
Распределение температуры не нормальное: из-за сезонности оно двугорбое и имеет
более короткие хвосты. К сезонному ряду сигма-правило нельзя применять напрямую,
корректнее анализировать остатки после снятия сезонности.
""")

# ===========================================================================
# Раздел 7. Дополнительная часть
# ===========================================================================
md(r"""
---
## Дополнительная часть. Лепестковая диаграмма (роза ветров)

Пункт 2 дополнительной части: построение диаграммы из списка
«трёхкомпонентная, карта, лица Чернова, поверхность, график с выделенными
областями, лепестковая (радарная)».

Выбрана роза ветров — лепестковая диаграмма в полярных координатах. Она
уместна содержательно: это канонический инструмент метеорологии, и, в отличие
от круговой диаграммы из пункта 2, показывает одновременно повторяемость
направлений и распределение скоростей по 16 румбам.
""")

code(r"""
speed_bins = [0, 2, 4, 6, 8, np.inf]
speed_labels = ["0–2", "2–4", "4–6", "6–8", "> 8"]

wind = df[["wind_dir", "wind_speed"]].copy()
wind["rumb"] = pd.cut((wind["wind_dir"] + 11.25) % 360,
                      bins=np.arange(0, 361, 22.5),
                      labels=RUMBS_16, right=False, include_lowest=True)
wind["grade"] = pd.cut(wind["wind_speed"], bins=speed_bins,
                       labels=speed_labels, right=False)

pivot = (pd.crosstab(wind["rumb"], wind["grade"])
         .reindex(RUMBS_16).fillna(0))
pivot_pct = pivot / len(wind) * 100

theta = np.deg2rad(np.arange(0, 360, 22.5))
width = np.deg2rad(22.5) * 0.92

fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"projection": "polar"})
ax.set_theta_zero_location("N")   # север вверху
ax.set_theta_direction(-1)        # по часовой стрелке, как принято в метеорологии

bottom = np.zeros(16)
colors = plt.cm.YlGnBu(np.linspace(0.35, 1.0, len(speed_labels)))
for label, color in zip(speed_labels, colors):
    values = pivot_pct[label].to_numpy()
    ax.bar(theta, values, width=width, bottom=bottom, color=color,
           edgecolor="white", lw=0.6, label=f"{label} м/с")
    bottom += values

ax.set_xticks(theta)
ax.set_xticklabels(RUMBS_16, fontsize=10)
ax.set_rlabel_position(112.5)
ax.grid(True, ls=":", alpha=0.6)
ax.set_title(f"Роза ветров, Пермь, 2024 (n = {len(df)} наблюдений)\n"
             "радиус — повторяемость, %", pad=24, fontsize=13)
ax.legend(title="Скорость ветра", loc="upper left",
          bbox_to_anchor=(1.06, 1.06), fontsize=9)

fig.savefig(FIG / "17_windrose.png")
plt.show()
""")

code(r"""
total_pct = pivot_pct.sum(axis=1).sort_values(ascending=False)
print("Повторяемость направлений, % (по убыванию):")
print(total_pct.round(2).head(6).to_string())

calm = (df["wind_speed"] < 0.5).mean() * 100
print(f"\nСредняя скорость ветра: {df['wind_speed'].mean():.2f} м/с")
print(f"Максимальная скорость:  {df['wind_speed'].max():.2f} м/с")
print(f"Доля штилей (< 0.5 м/с): {calm:.1f} %")

strong = wind[wind["wind_speed"] >= 6]["rumb"].value_counts().head(3)
print("\nРумбы с наибольшим числом сильных ветров (≥ 6 м/с):")
print(strong.to_string())
""")

md(r"""
Интерпретация розы ветров. Диаграмма показывает преобладающие направления
переноса воздушных масс в районе Перми. Лепестки наибольшей длины отвечают
господствующим румбам, а распределение цвета внутри лепестка показывает, какие
скорости характерны для каждого направления: сильные ветры обычно связаны с
конкретными румбами, поскольку определяются типовыми траекториями циклонов.

Про штили: при скорости, близкой к нулю,
направление ветра физически не определено, и такие случаи формально
распределяются по румбам, слегка искажая картину. Их доля указана выше
отдельно, чтобы интерпретация оставалась корректной.

Роза ветров дополняет круговую диаграмму из пункта 2: та отвечает только на
вопрос «откуда чаще дует», а лепестковая — ещё и «насколько сильно дует с
каждого направления».
""")

# ===========================================================================
# Раздел 8. Итоги
# ===========================================================================
md(r"""
---
## Итоги работы

Все пункты задания закрыты: данные загружены и приведены к нужным типам
и построены четыре обязательные диаграммы — линейная, столбчатая, круговая и
скрипичная (пункт 2), проведён качественный анализ предметной области
(пункт 3), пропуски созданы прореживанием и заполнены пятью методами
(пункт 4), недостоверные значения найдены правилом 2σ/3σ с проверкой по
box-plot (пункт 5). Дополнительно — собственный набор данных и лепестковая
диаграмма. Целевая оценка 9 баллов.

Блок «+2 балла» (пропуски по специальным схемам и метрики MAE/RMSE/MAPE/MASE)
в объём работы не входил; при необходимости он достраивается в разделе 4.6,
где уже собрана таблица отклонений по всем методам.

### Содержательные результаты

1. Годовая амплитуда температуры превышает 63 °C — климат резко
   континентальный.
2. Амплитуда суточного хода летом в несколько раз больше зимней из-за различий
   в радиационном балансе.
3. Температура и относительная влажность связаны обратной зависимостью:
   тёплый воздух удерживает больше пара, поэтому дневной нагрев снижает
   относительную влажность.
4. Точность методов восстановления различается более чем на порядок: от долей
   градуса у кубического сплайна до 3 °C у заполнения константой.
5. Многочлен Лагранжа по 24 узлам неприменим из-за эффекта Рунге: его степень
   растёт вместе с числом узлов. Кубический сплайн этого недостатка лишён,
   потому что составлен из отдельных кусков третьей степени.
6. Распределение температуры за год не нормальное: двугорбое, с укороченными
   хвостами, поэтому сигма-правило применимо к такому ряду лишь с оговорками.
""")

# ===========================================================================
nb = {
    "cells": [
        {
            "cell_type": kind,
            "id": f"cell-{n:02d}",          # nbformat 4.5+ требует идентификатор
            "metadata": {},
            "source": text.splitlines(keepends=True),
            **({"execution_count": None, "outputs": []} if kind == CODE else {}),
        }
        for n, (kind, text) in enumerate(cells)
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python (ВШЭ ML ДЗ1)",
            "language": "python",
            "name": "hse_ml_hw1",
        },
        "language_info": {"name": "python", "version": "3.11.15"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "hw1_perm_weather.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

n_code = sum(1 for k, _ in cells if k == CODE)
print(f"Записан {out}")
print(f"Ячеек всего: {len(cells)} (код: {n_code}, markdown: {len(cells) - n_code})")
