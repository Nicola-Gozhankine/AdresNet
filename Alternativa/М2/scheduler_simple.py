"""
scheduler_simple.py

Простой запуск эволюции с одним островом, без миграции и сложных фаз.
Цель: проверить работу всех модулей и увидеть, как растёт точность.

Параметры:
- популяция: 200
- поколения: 100
- целевой показатель: 0.6 (останавливаемся при достижении)
- фитнес: фаза 1 (без штрафа за сложность)
- логирование в CSV-файл
"""

import random
import csv
import os
import sys

# Подключаем наши модули (предполагается, что они в той же директории)
from genotype import random_genotype, mutate, crossover
from fitness import fitness
from island import Island

# Константы
POP_SIZE = 200
GENERATIONS = 100
TARGET = 0.6          # останавливаемся, если penalized fitness >= TARGET
PHASE = 1             # первая фаза (без штрафа за сложность)
LOG_FILE = "evolution_log.csv"

def main():
    print("Запуск простой эволюции с одним островом")
    print(f"Популяция: {POP_SIZE}, поколений: {GENERATIONS}, целевой показатель: {TARGET}")

    # Создаём случайную начальную популяцию
    initial_pop = [random_genotype() for _ in range(POP_SIZE)]

    # Создаём остров
    island = Island(
        population=initial_pop,
        fitness_func=fitness,
        phase=PHASE,
        elite_size=10,            # сохраняем 10 лучших
        tournament_size=5,
        mutation_rate=0.1,
        crossover_rate=0.8,
        genotype_module=sys.modules[__name__]  # передаём текущий модуль (где есть mutate/crossover)
    )

    # Подготовка лога
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_raw", "best_pen", "avg_raw", "avg_pen", "min_raw"])

    best_pen_global = 0.0

    for gen in range(GENERATIONS):
        # Одно поколение эволюции
        island.evolve_one_generation()

        # Получаем статистику
        best_list = island.get_best(5)  # топ-5
        best_raw = best_list[0][1]
        best_pen = best_list[0][2]

        # Средние значения по всей популяции (вычисляем через оценку всех)
        raw_vals = []
        pen_vals = []
        for g in island.population:
            r, p = island._evaluate(g)
            raw_vals.append(r)
            pen_vals.append(p)
        avg_raw = sum(raw_vals) / len(raw_vals)
        avg_pen = sum(pen_vals) / len(pen_vals)
        min_raw = min(raw_vals)

        # Логируем
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([gen, best_raw, best_pen, avg_raw, avg_pen, min_raw])

        print(f"Поколение {gen:3d}: best_raw={best_raw:.4f}, best_pen={best_pen:.4f}, avg_raw={avg_raw:.4f}")

        # Обновляем глобальный лучший
        if best_pen > best_pen_global:
            best_pen_global = best_pen
            # Сохраняем лучший генотип отдельно
            with open("best_genotype.json", "w") as f:
                import json
                json.dump(best_list[0][0], f)

        # Проверка на достижение цели
        if best_pen >= TARGET:
            print(f"✅ Достигнут целевой показатель {TARGET} на поколении {gen}")
            break

    print("Эволюция завершена.")
    print(f"Лучший фитнес (penalized): {best_pen_global:.4f}")
    print(f"Лог сохранён в {LOG_FILE}")

if __name__ == "__main__":
    main()