"""Core methods for evaluating investment projects under uncertainty."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real


def _as_finite_numbers(name: str, values: Sequence[Real]) -> list[float]:
    """Validate a numeric sequence and return a plain list of floats."""
    if not values:
        raise ValueError(f"{name} не должен быть пустым")

    result: list[float] = []
    for value in values:
        if not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{name} должен содержать только конечные числа")
        result.append(float(value))
    return result


def _validate_trapezoid(values: Sequence[Real]) -> tuple[float, float, float, float]:
    """Validate the ordered points of a trapezoidal fuzzy number."""
    points = _as_finite_numbers("Точки нечёткого NPV", values)
    if len(points) != 4:
        raise ValueError("Нечёткий NPV должен состоять ровно из четырёх точек")
    if points != sorted(points):
        raise ValueError("Ожидается порядок npv1 <= npv2 <= npv3 <= npv4")
    return tuple(points)


def risk_class(value: float) -> str:
    """Return the qualitative class for a non-negative relative-risk value."""
    if not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(
            "Коэффициент риска должен быть конечным неотрицательным числом"
        )
    if value < 0.15:
        return "низкий"
    if value <= 0.30:
        return "средний"
    return "высокий"


@dataclass(frozen=True)
class BudgetEfficiencyResult:
    """Result of the budget-efficiency calculation."""

    npv: float
    pv_incomes: float
    pv_supports: float
    pi: float
    payback_t: int | None
    pv_net_by_year: list[float]


def budget_efficiency(
    years: Sequence[int],
    incomes: Sequence[Real],
    supports: Sequence[Real],
    rate: float,
    base_year: int | None = None,
) -> BudgetEfficiencyResult:
    """Calculate discounted budget NPV, profitability index and payback year."""
    if not years:
        raise ValueError("Список years не должен быть пустым")
    if len(years) != len(incomes) or len(years) != len(supports):
        raise ValueError("years, incomes и supports должны быть одной длины")
    if any(not isinstance(year, int) for year in years):
        raise ValueError("years должен содержать целые годы")
    if any(current <= previous for previous, current in pairwise(years)):
        raise ValueError("Годы должны идти в строго возрастающем порядке")
    if not isinstance(rate, Real) or not math.isfinite(float(rate)) or rate <= -1:
        raise ValueError("Ставка дисконтирования должна быть конечным числом больше -1")

    checked_incomes = _as_finite_numbers("incomes", incomes)
    checked_supports = _as_finite_numbers("supports", supports)
    base_year = years[0] if base_year is None else base_year
    if not isinstance(base_year, int):
        raise TypeError("base_year должен быть целым годом")

    pv_incomes = 0.0
    pv_supports = 0.0
    pv_net_by_year: list[float] = []
    cumulative = 0.0
    payback_t: int | None = None

    for year, income, support in zip(years, checked_incomes, checked_supports):
        t = year - base_year
        discount = 1 / (1 + rate) ** t
        pv_income = income * discount
        pv_support = support * discount
        pv_net = pv_income - pv_support

        pv_incomes += pv_income
        pv_supports += pv_support
        pv_net_by_year.append(pv_net)
        cumulative += pv_net
        if payback_t is None and cumulative >= 0:
            payback_t = t

    return BudgetEfficiencyResult(
        npv=pv_incomes - pv_supports,
        pv_incomes=pv_incomes,
        pv_supports=pv_supports,
        pi=pv_incomes / pv_supports if pv_supports != 0 else float("inf"),
        payback_t=payback_t,
        pv_net_by_year=pv_net_by_year,
    )


@dataclass(frozen=True)
class ScenarioResult:
    """Expected value and risk metrics for a discrete scenario distribution."""

    expected_npv: float
    variance: float
    sigma: float
    variation: float
    risk: str


def scenario_method(scenarios: Sequence[tuple[Real, Real]]) -> ScenarioResult:
    """Calculate expected NPV and relative risk for weighted scenarios."""
    if not scenarios:
        raise ValueError("Нужен хотя бы один сценарий")

    checked: list[tuple[float, float]] = []
    for scenario in scenarios:
        if len(scenario) != 2:
            raise ValueError("Каждый сценарий должен быть парой (NPV, вероятность)")
        npv, probability = scenario
        if not isinstance(npv, Real) or not math.isfinite(float(npv)):
            raise ValueError("NPV сценария должен быть конечным числом")
        if (
            not isinstance(probability, Real)
            or not math.isfinite(float(probability))
            or probability < 0
        ):
            raise ValueError(
                "Вероятности должны быть конечными неотрицательными числами"
            )
        checked.append((float(npv), float(probability)))

    probability_sum = sum(probability for _, probability in checked)
    if not math.isclose(probability_sum, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Сумма вероятностей сценариев должна быть равна 1")

    expected = sum(probability * npv for npv, probability in checked)
    variance = sum(probability * (npv - expected) ** 2 for npv, probability in checked)
    sigma = math.sqrt(variance)
    variation = sigma / abs(expected) if expected != 0 else float("inf")

    return ScenarioResult(
        expected_npv=expected,
        variance=variance,
        sigma=sigma,
        variation=variation,
        risk="не определён" if math.isinf(variation) else risk_class(variation),
    )


@dataclass(frozen=True)
class FuzzyNpvResult:
    """Possibility that a trapezoidal fuzzy NPV is non-negative."""

    possibility: float
    acceptable: bool


def fuzzy_npv(
    npv1: float,
    npv2: float,
    npv3: float,
    npv4: float,
    threshold: float = 0.8,
) -> FuzzyNpvResult:
    """Estimate Poss(NPV >= 0) for an ordered trapezoidal fuzzy number."""
    npv1, npv2, npv3, npv4 = _validate_trapezoid((npv1, npv2, npv3, npv4))
    if (
        not isinstance(threshold, Real)
        or not math.isfinite(float(threshold))
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Порог приемлемости должен находиться в диапазоне от 0 до 1")

    if npv1 >= 0:
        possibility = 1.0
    elif npv4 <= 0:
        possibility = 0.0
    elif npv2 >= 0 or npv3 >= 0:
        possibility = 1.0
    else:
        possibility = npv4 / (npv4 - npv3)

    return FuzzyNpvResult(
        possibility=possibility,
        acceptable=possibility >= threshold,
    )


@dataclass(frozen=True)
class ChiuParkResult:
    """Point estimate and dispersion index for the Chiu-Park method."""

    m: float
    v: float
    relative_risk: float
    risk: str


def chiu_park(
    npv1: float,
    npv2: float,
    npv3: float,
    npv4: float,
    lam: float = 1.0,
) -> ChiuParkResult:
    """Collapse a trapezoidal fuzzy NPV into return and risk indicators."""
    npv1, npv2, npv3, npv4 = _validate_trapezoid((npv1, npv2, npv3, npv4))
    if not isinstance(lam, Real) or not math.isfinite(float(lam)) or lam < 0:
        raise ValueError("lam должен быть конечным неотрицательным числом")

    m = (npv1 + 2 * lam * (npv2 + npv3) / 2 + npv4) / (2 + 2 * lam)
    v = (npv4 - npv1) / 4
    relative_risk = v / abs(m) if m != 0 else float("inf")

    return ChiuParkResult(
        m=m,
        v=v,
        relative_risk=relative_risk,
        risk="не определён" if math.isinf(relative_risk) else risk_class(relative_risk),
    )
