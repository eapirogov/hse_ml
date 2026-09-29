"""Вспомогательные функции для задания 2: метрики и рекурсивный прогноз.

Вынесено отдельным модулем по тому же принципу, что и `hw_utils.py` для
задания 1: ячейки ноутбука остаются читаемыми, а логика прогнозирования
проверяется без запуска Jupyter.
"""

from __future__ import annotations

import numpy as np

# Горизонт прогноза строится «в отрыве» от теста: модель не видит ни одного
# истинного значения из тестовой выборки. Это ключевое условие пункта 1
# задания — прогнозируем 20 % значений по первым 80 %.


def mae(truth: np.ndarray, pred: np.ndarray) -> float:
    """Средняя абсолютная ошибка."""
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    return float(np.mean(np.abs(truth - pred)))


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    """Корень из средней квадратической ошибки."""
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def r2(truth: np.ndarray, pred: np.ndarray) -> float:
    """Коэффициент детерминации R² = 1 − SS_res / SS_tot.

    Доля дисперсии зависимой переменной,
    объяснённая моделью. На тестовой выборке может оказаться отрицательным —
    это значит, что модель хуже простой константы (среднего теста).
    """
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
    return float(1.0 - ss_res / ss_tot)


def metrics_row(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Три метрики одной строкой — для сводных таблиц сравнения моделей."""
    return {"MAE": mae(truth, pred), "RMSE": rmse(truth, pred), "R2": r2(truth, pred)}


def moving_average_forecast(history: np.ndarray, horizon: int, k: int) -> np.ndarray:
    """Прогноз скользящим средним (k-NN по k предшествующим точкам).

    Реализует формулу y = (сумма k предыдущих значений) / k.
    На горизонте > 1 шага работает рекурсивно: полученный прогноз добавляется
    в историю и служит опорой для следующего
    шага. Поэтому ряд прогноза быстро «выполаживается» к среднему последних
    значений: собственные оценки усредняются сами с собой.
    """
    hist = list(np.asarray(history, float))
    out = []
    for _ in range(horizon):
        step = float(np.mean(hist[-k:]))
        out.append(step)
        hist.append(step)
    return np.asarray(out)


def make_lag_matrix(series: np.ndarray, n_lags: int):
    """Матрица лагов: строка i — n_lags предыдущих значений, цель — значение i.

    Используется там, где по условию модель должна опираться на предыдущие
    значения искомого параметра (авторегрессионная постановка: прошлые
    значения зависимой переменной выступают как регрессоры).
    """
    series = np.asarray(series, float)
    x, y = [], []
    for i in range(n_lags, len(series)):
        x.append(series[i - n_lags:i])
        y.append(series[i])
    return np.asarray(x), np.asarray(y)


def recursive_forecast(model, history: np.ndarray, horizon: int, n_lags: int,
                       scaler=None) -> np.ndarray:
    """Рекурсивный прогноз обученной модели по её же предыдущим значениям.

    Прогнозные значения используем как реальные для следующего шага.
    На каждом шаге берём последние n_lags значений истории
    (сначала настоящих, затем всё больше собственных прогнозов), получаем
    оценку и дописываем её в историю.
    """
    hist = list(np.asarray(history, float))
    out = []
    for _ in range(horizon):
        window = np.asarray(hist[-n_lags:], float).reshape(1, -1)
        if scaler is not None:
            window = scaler.transform(window)
        step = float(np.ravel(model.predict(window))[0])
        out.append(step)
        hist.append(step)
    return np.asarray(out)


def recursive_forecast_exog(model, history: np.ndarray, exog_future: np.ndarray,
                            n_lags: int, scaler=None) -> np.ndarray:
    """То же, но к лагам искомого параметра добавлен второй параметр.

    Для многопараметрической модели нужен прогноз каждого из параметров.
    `exog_future` — уже спрогнозированные значения второго
    параметра на весь горизонт, истинных тестовых значений модель не видит.
    """
    hist = list(np.asarray(history, float))
    exog_future = np.asarray(exog_future, float)
    out = []
    for step_i in range(len(exog_future)):
        window = np.r_[np.asarray(hist[-n_lags:], float), exog_future[step_i]]
        window = window.reshape(1, -1)
        if scaler is not None:
            window = scaler.transform(window)
        step = float(np.ravel(model.predict(window))[0])
        out.append(step)
        hist.append(step)
    return np.asarray(out)
