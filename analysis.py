"""
Сводный анализ пяти проектов: прогоняем каждый через все четыре метода,
собираем результаты в таблицу и строим рейтинг по методу Чиу-Парка.
"""

import pandas as pd

from investment_methods import (
    budget_efficiency,
    chiu_park,
    fuzzy_npv,
    scenario_method,
)
from projects_data import PROJECTS


def analyze_project(project: dict) -> dict:
    """
    Считает один проект всеми четырьмя методами и возвращает готовые
    объекты-результаты в одном словаре.
    """
    b = project["budget"]

    budget = budget_efficiency(
        years=b["years"],
        incomes=b["incomes"],
        supports=b["supports"],
        rate=b["rate"],
        base_year=b["base_year"],
    )
    scen = scenario_method(project["scenarios"])
    fuzzy = fuzzy_npv(*project["fuzzy"])
    cp = chiu_park(*project["fuzzy"])

    return {"budget": budget, "scenario": scen, "fuzzy": fuzzy, "chiu_park": cp}


def build_summary_table(projects: dict = PROJECTS) -> pd.DataFrame:
    """
    Собирает сводную таблицу ключевых показателей по всем проектам —
    по одной строке на проект, по колонке на показатель.
    """
    rows = []
    for code, project in projects.items():
        r = analyze_project(project)
        rows.append(
            {
                "Проект": code,
                "NPV бюджета": round(r["budget"].npv, 2),
                "PI": round(r["budget"].pi, 2),
                "E(NPV) сценарии": round(r["scenario"].expected_npv, 2),
                "Коэф. вариации V": round(r["scenario"].variation, 4),
                "Poss(NPV≥0)": round(r["fuzzy"].possibility, 3),
                "M(NPV) Чиу-Парк": round(r["chiu_park"].m, 2),
                "V/M": round(r["chiu_park"].relative_risk, 4),
                "Класс риска": r["chiu_park"].risk,
            }
        )
    return pd.DataFrame(rows)


def build_ranking(projects: dict = PROJECTS) -> pd.DataFrame:
    """
    Строит рейтинг проектов по методу Чиу-Парка: сортируем по убыванию
    M(NPV) (чем выше — тем эффективнее) и добавляем колонку ранга.
    """
    rows = []
    for code, project in projects.items():
        r = analyze_project(project)
        cp = r["chiu_park"]
        rows.append(
            {
                "Проект": code,
                "M(NPV)": round(cp.m, 2),
                "V(NPV)": round(cp.v, 2),
                "V/M": round(cp.relative_risk, 4),
                "Класс риска": cp.risk,
            }
        )

    ranking = pd.DataFrame(rows).sort_values("M(NPV)", ascending=False)
    ranking.insert(0, "Ранг", range(1, len(ranking) + 1))
    return ranking.reset_index(drop=True)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("=== Сводная таблица показателей по всем проектам ===\n")
    print(build_summary_table().to_string(index=False))

    print("\n\n=== Рейтинг проектов по методу Чиу-Парка ===\n")
    print(build_ranking().to_string(index=False))
