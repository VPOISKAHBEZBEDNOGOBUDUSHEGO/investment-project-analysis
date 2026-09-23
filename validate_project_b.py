"""Command-line validation of the calculation core against project B."""

from __future__ import annotations

import math

from investment_methods import budget_efficiency, chiu_park, fuzzy_npv, scenario_method


def check(name: str, got: float, expected: float, tolerance: float = 0.5) -> bool:
    """Print one comparison and return whether it is within tolerance."""
    ok = math.isclose(got, expected, rel_tol=0, abs_tol=tolerance)
    status = "OK" if ok else "РАСХОЖДЕНИЕ"
    print(f"  [{status}] {name}: получено {got:.2f}, эталон {expected:.2f}")
    return ok


def run_validation() -> bool:
    """Run all reference checks and return a single success flag."""
    all_ok = True

    print("Метод 1 - ВК 477 (бюджетная эффективность)")
    budget = budget_efficiency(
        years=[2022, 2023, 2024, 2025, 2026],
        incomes=[9741.70, 11578.20, 12358.50, 13939.852586609888, 15723.549794587885],
        supports=[2643.80, 1807.90, 367.20, 0.0, 0.0],
        rate=0.15,
        base_year=2021,
    )
    all_ok &= check("NPV бюджета", budget.npv, 37231.850940729179)
    all_ok &= check("PV доходов", budget.pv_incomes, 41139.279559123199)
    all_ok &= check("PV поддержки", budget.pv_supports, 3907.4286183940176)
    all_ok &= check("Индекс доходности PI", budget.pi, 10.528478848074709, 0.01)
    all_ok &= budget.payback_t == 1

    print("\nМетод 2 - метод сценариев")
    scenario = scenario_method(
        [
            (29003.995028904545, 0.25),
            (37231.850940729179, 0.50),
            (43402.742874597658, 0.25),
        ]
    )
    all_ok &= check("E(NPV)", scenario.expected_npv, 36717.60994624014)
    all_ok &= check("СКО", scenario.sigma, 5116.6332916180918)
    all_ok &= check(
        "Коэффициент вариации", scenario.variation, 0.13935093539883392, 0.001
    )

    points = (
        26947.031050948382,
        37231.850940729179,
        39185.565249926192,
        47413.421161750826,
    )
    print("\nМетод 3 - нечёткий NPV")
    fuzzy = fuzzy_npv(*points)
    all_ok &= check("Poss(NPV >= 0)", fuzzy.possibility, 1.0, 0.0001)

    print("\nМетод 4 - метод Чиу-Парка")
    cp = chiu_park(*points)
    all_ok &= check("M(NPV)", cp.m, 37694.46710083865)
    all_ok &= check("V(NPV)", cp.v, 5116.597527700611)
    all_ok &= check("V/M", cp.relative_risk, 0.13573868849274098, 0.001)

    print()
    print(
        "ИТОГ: все показатели сошлись с эталоном."
        if all_ok
        else "ИТОГ: есть расхождения."
    )
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if run_validation() else 1)
