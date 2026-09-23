import math

import pytest

from investment_methods import (
    budget_efficiency,
    chiu_park,
    fuzzy_npv,
    risk_class,
    scenario_method,
)


def test_project_b_matches_reference_values():
    budget = budget_efficiency(
        years=[2022, 2023, 2024, 2025, 2026],
        incomes=[9741.70, 11578.20, 12358.50, 13939.852586609888, 15723.549794587885],
        supports=[2643.80, 1807.90, 367.20, 0.0, 0.0],
        rate=0.15,
        base_year=2021,
    )
    assert budget.npv == pytest.approx(37231.850940729179)
    assert budget.pv_incomes == pytest.approx(41139.279559123199)
    assert budget.pv_supports == pytest.approx(3907.4286183940176)
    assert budget.pi == pytest.approx(10.528478848074709)
    assert budget.payback_t == 1

    scenarios = scenario_method(
        [
            (29003.995028904545, 0.25),
            (37231.850940729179, 0.50),
            (43402.742874597658, 0.25),
        ]
    )
    assert scenarios.expected_npv == pytest.approx(36717.60994624014)
    assert scenarios.sigma == pytest.approx(5116.6332916180918)
    assert scenarios.variation == pytest.approx(0.13935093539883392)
    assert scenarios.risk == "низкий"

    points = (
        26947.031050948382,
        37231.850940729179,
        39185.565249926192,
        47413.421161750826,
    )
    fuzzy = fuzzy_npv(*points)
    assert fuzzy.possibility == 1.0
    assert fuzzy.acceptable is True

    cp = chiu_park(*points)
    assert cp.m == pytest.approx(37694.46710083865)
    assert cp.v == pytest.approx(5116.597527700611)
    assert cp.relative_risk == pytest.approx(0.13573868849274098)
    assert cp.risk == "низкий"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "низкий"),
        (0.149, "низкий"),
        (0.15, "средний"),
        (0.30, "средний"),
        (0.301, "высокий"),
    ],
)
def test_risk_class_boundaries(value, expected):
    assert risk_class(value) == expected


@pytest.mark.parametrize("value", [-1, math.nan, math.inf])
def test_risk_class_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        risk_class(value)


def test_budget_efficiency_validates_lengths_and_year_order():
    with pytest.raises(ValueError, match="одной длины"):
        budget_efficiency([2024, 2025], [10], [2, 3], 0.1)
    with pytest.raises(ValueError, match="возрастающем"):
        budget_efficiency([2025, 2024], [10, 11], [2, 3], 0.1)


def test_scenario_method_validates_probability_distribution():
    with pytest.raises(ValueError, match="равна 1"):
        scenario_method([(10, 0.4), (20, 0.4)])
    with pytest.raises(ValueError, match="неотрицательными"):
        scenario_method([(10, 1.1), (20, -0.1)])


def test_scenario_relative_risk_is_non_negative_for_negative_expected_npv():
    result = scenario_method([(-20, 0.5), (-10, 0.5)])
    assert result.expected_npv == -15
    assert result.variation > 0


def test_fuzzy_methods_require_ordered_points():
    with pytest.raises(ValueError, match="порядок"):
        fuzzy_npv(0, 10, 5, 20)
    with pytest.raises(ValueError, match="порядок"):
        chiu_park(0, 10, 5, 20)


def test_fuzzy_threshold_and_lambda_are_validated():
    with pytest.raises(ValueError, match="диапазоне"):
        fuzzy_npv(-10, -5, 5, 10, threshold=1.1)
    with pytest.raises(ValueError, match="lam"):
        chiu_park(-10, -5, 5, 10, lam=-1)
