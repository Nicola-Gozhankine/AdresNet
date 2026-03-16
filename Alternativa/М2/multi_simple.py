"""
multi_parity.py

Улучшенная версия многоконтинентальной эволюции для задачи чётности (parity).
- Только одна задача (чётность), никакой мультизадачности.
- Возможность загружать лучшие генотипы из файла и внедрять их в начальные популяции.
- Увеличенные параметры: популяция 500, поколений 500 (настраиваются).
- Несколько континентов (процессов), каждый с несколькими островами.
- Сохранение общего лога лучших моделей.
"""

import multiprocessing as mp
import random
import json
import os
import time
from typing import List, Tuple, Any

# Импортируем наши модули (должны быть в той же папке)
from island import Island
from genotype import random_genotype, mutate, crossover
from fitness import fitness   # fitness(genotype, phase) возвращает (raw, pen)

# ======================== НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ========================
NUM_CONTINENTS = 4                  # число континентов (процессов)
ISLANDS_PER_CONTINENT = 5           # островов на континенте
POP_SIZE = 500                       # размер популяции на острове
GENERATIONS = 500                    # число поколений
PHASE = 1                            # первая фаза (без штрафа за сложность)

# Инъекция лога
INJECT_LOG = True                    # загружать ли предыдущие лучшие модели
LOG_FILE = "final_best.json"          # файл с лучшими моделями (если есть)
INJECT_COUNT = 20                     # сколько лучших моделей добавить в каждую начальную популяцию

# Логирование
LOG_DIR = "continent_logs_parity"    # папка для логов континентов
FINAL_LOG = "final_best_parity.json" # итоговый лог
FINAL_LOG_SIZE = 100                  # сколько лучших хранить

# ======================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ========================
def load_best_genotypes(filename: str, max_count: int) -> List[Any]:
    """Загружает до max_count лучших генотипов из файла (формат: список [fitness, genotype])."""
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as f:
        data = json.load(f)
    # Ожидаем структуру: [[fitness, genotype], ...]
    # Сортируем на всякий случай и берём первые max_count
    data.sort(reverse=True, key=lambda x: x[0])
    return [entry[1] for entry in data[:max_count]]

# ======================== ФУНКЦИЯ КОНТИНЕНТА ========================
def run_continent(continent_id: int, inject_genotypes: List[Any]):
    """Запускает эволюцию на одном континенте."""
    print(f"[Континент {continent_id}] Запуск с POP_SIZE={POP_SIZE}, GENERATIONS={GENERATIONS}")
    islands = []

    # Создаём острова
    for i in range(ISLANDS_PER_CONTINENT):
        # Формируем начальную популяцию: сначала инъекция (если есть), потом случайные
        pop = []
        if inject_genotypes:
            # Берём случайную выборку из инъекционных генотипов
            sample_size = min(INJECT_COUNT, len(inject_genotypes))
            pop.extend(random.sample(inject_genotypes, sample_size))
        # Добиваем до POP_SIZE случайными
        while len(pop) < POP_SIZE:
            pop.append(random_genotype())
        random.shuffle(pop)

        island = Island(
            population=pop,
            fitness_func=fitness,
            phase=PHASE,
            elite_size=POP_SIZE // 20,
            tournament_size=5,
            mutation_rate=0.1,
            crossover_rate=0.8,
            genotype_module=None   # использует стандартные mutate/crossover из genotype
        )
        islands.append(island)

    # Основной цикл эволюции
    for gen in range(GENERATIONS):
        for island in islands:
            island.evolve_one_generation()

        # Каждые 50 поколений выводим прогресс (берём лучший с первого острова)
        if gen % 50 == 0:
            best_list = islands[0].get_best(1)
            if best_list:
                best_raw, best_pen = best_list[0][1], best_list[0][2]
                print(f"[Континент {continent_id}] Поколение {gen}: лучший raw={best_raw:.4f}, pen={best_pen:.4f}")

    # Собираем лучшие генотипы со всех островов
    continent_best = []
    for island in islands:
        top5 = island.get_best(5)   # (genotype, raw, pen)
        for g, raw, pen in top5:
            continent_best.append((pen, g))  # для лога храним (fitness, genotype)

    # Сортируем и оставляем, например, 50 лучших с континента
    continent_best.sort(reverse=True, key=lambda x: x[0])
    continent_best = continent_best[:50]

    # Сохраняем лог континента
    os.makedirs(LOG_DIR, exist_ok=True)
    filename = os.path.join(LOG_DIR, f"continent_{continent_id}.json")
    with open(filename, 'w') as f:
        json.dump(continent_best, f)

    print(f"[Континент {continent_id}] Завершён. Лучший фитнес: {continent_best[0][0] if continent_best else 0:.4f}")

# ======================== ОСНОВНОЙ КООРДИНАТОР ========================
def main():
    print("🚀 Запуск улучшенной многоконтинентальной эволюции (только parity)")
    print(f"Континентов: {NUM_CONTINENTS}, островов/континент: {ISLANDS_PER_CONTINENT}")
    print(f"Популяция: {POP_SIZE}, поколений: {GENERATIONS}")

    # Загружаем лучшие модели из предыдущего запуска (если нужно)
    inject_list = []
    if INJECT_LOG and os.path.exists(LOG_FILE):
        inject_list = load_best_genotypes(LOG_FILE, INJECT_COUNT * NUM_CONTINENTS * ISLANDS_PER_CONTINENT)
        print(f"Загружено {len(inject_list)} генотипов для инъекции из {LOG_FILE}")

    # Запускаем континенты в отдельных процессах
    processes = []
    for cid in range(NUM_CONTINENTS):
        p = mp.Process(target=run_continent, args=(cid, inject_list))
        processes.append(p)
        p.start()

    # Ждём завершения всех
    for p in processes:
        p.join()

    # Собираем все логи континентов
    all_best = []
    for cid in range(NUM_CONTINENTS):
        filename = os.path.join(LOG_DIR, f"continent_{cid}.json")
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                all_best.extend(data)

    # Сортируем и оставляем лучшие FINAL_LOG_SIZE
    all_best.sort(reverse=True, key=lambda x: x[0])
    final_best = all_best[:FINAL_LOG_SIZE]

    # Сохраняем финальный лог
    with open(FINAL_LOG, 'w') as f:
        json.dump(final_best, f)

    print(f"\n✅ Все континенты завершены. Лучший общий фитнес: {final_best[0][0] if final_best else 0:.4f}")
    print(f"Финальный лог сохранён в {FINAL_LOG}")

if __name__ == "__main__":
    main()