"""
multi_continent.py

Многоконтинентальная эволюционная система.
Создаёт несколько континентов (процессов), на каждом континенте несколько островов.
Континенты эволюционируют параллельно, обмениваясь лучшими особями через общий лог.
После серии запусков (трендов) проводится битва континентов, в которой худшие острова отсеиваются,
а лучшие получают больше ресурсов. В конце остаются 100 лучших моделей в финальном логе.

Архитектура:
- Континент = отдельный процесс (multiprocessing)
- На континенте 5 островов (Island из island.py)
- Каждый континент ведёт свой лог лучших особей
- Глобальный лог (JSON) собирает лучших со всех континентов
- Битва континентов: сравниваем лучшие фитнесы, сокращаем слабые острова
"""

import multiprocessing as mp
import random
import json
import os
import time
from typing import List, Dict, Any

# Импортируем наши модули
from island import Island
from genotype import random_genotype, mutate, crossover
from fitness import fitness

# Конфигурация континентов
CONTINENTS = 5                 # число континентов
ISLANDS_PER_CONTINENT = 5      # островов на континенте

# Фазы (тренды) - список кортежей (популяция, поколения)
TRENDS = [
    (100, 100),    # тренд 1: разогрев
    (200, 500),    # тренд 2: средний
    (500, 500),    # тренд 3: усиленный
    (1000, 100),   # тренд 4: с использованием лога (популяция большая, но мало поколений)
    (2000, 500)    # тренд 5: финальный рывок
]

# Параметры битвы
BATTLE_ELITE_RATIO = 0.3       # доля лучших островов, которые получают усиление
BATTLE_POP_BOOST = 2           # увеличение популяции для лучших островов
BATTLE_GEN_BOOST = 1.5         # увеличение числа поколений (коэффициент)

# Финальный лог
FINAL_LOG = "final_best.json"
FINAL_LOG_SIZE = 100            # храним 100 лучших

# Временные файлы для обмена между процессами
GLOBAL_LOG = "global_log.json"
GLOBAL_LOG_SIZE = 500

# ----------------------------------------------------------------------
# Класс континента (запускается в отдельном процессе)
# ----------------------------------------------------------------------

class Continent:
    def __init__(self, continent_id, trend_index, initial_log=None):
        self.id = continent_id
        self.trend_index = trend_index
        self.pop_size, self.generations = TRENDS[trend_index]
        self.islands = []
        self.log = initial_log if initial_log else []  # лучшие особи с континента
        self.phase = 1  # первая фаза (без штрафа сложности)

        # Создаём острова
        for i in range(ISLANDS_PER_CONTINENT):
            # Начальная популяция: случайная + если есть лог, добавляем оттуда
            pop = []
            if self.log:
                # Берём несколько случайных из лога
                samples = random.sample(self.log, min(10, len(self.log)))
                for g in samples:
                    pop.append(g)
            # Добиваем случайными
            while len(pop) < self.pop_size:
                pop.append(random_genotype())
            random.shuffle(pop)

            island = Island(
                population=pop,
                fitness_func=fitness,
                phase=self.phase,
                elite_size=max(1, self.pop_size // 20),
                tournament_size=5,
                mutation_rate=0.1,
                crossover_rate=0.8,
                genotype_module=None  # используем стандартные mutate/crossover из genotype
            )
            self.islands.append(island)

    def run_trend(self):
        """Запускает эволюцию на всех островах в течение self.generations поколений."""
        for gen in range(self.generations):
            for island in self.islands:
                island.evolve_one_generation()

            # Каждые 10 поколений собираем лучших с островов в лог континента
            if gen % 10 == 0:
                for island in self.islands:
                    best_list = island.get_best(3)  # топ-3 с острова
                    for g, raw, pen in best_list:
                        # Сохраняем в лог континента (сырой генотип)
                        self._add_to_log(g, pen)

            # Выводим прогресс (можно отключить для тишины)
            if gen % 20 == 0:
                best_global = self.get_best_global()
                print(f"Континент {self.id}, тренд {self.trend_index}, поколение {gen}: best={best_global[1]:.4f}")

        # После завершения тренда сохраняем лог континента в файл
        self._save_log()

    def _add_to_log(self, genotype, fitness_val):
        """Добавляет генотип в лог континента, если он лучше худшего."""
        self.log.append((fitness_val, genotype))
        self.log.sort(reverse=True, key=lambda x: x[0])
        self.log = self.log[:GLOBAL_LOG_SIZE]

    def _save_log(self):
        """Сохраняет лог континента во временный файл."""
        filename = f"continent_{self.id}_log.json"
        with open(filename, 'w') as f:
            json.dump(self.log, f)

    def get_best_global(self):
        """Возвращает лучшую особь на континенте (глобально по всем островам)."""
        best_overall = None
        best_fit = -1
        for island in self.islands:
            b = island.get_best(1)[0]
            if b[2] > best_fit:
                best_fit = b[2]
                best_overall = b
        return best_overall

    def inject_log(self, global_log):
        """Внедряет глобальный лог в популяции островов (заменяет худших)."""
        if not global_log:
            return
        # Берём случайную выборку из глобального лога
        migrants = random.sample([g for _, g in global_log], min(10, len(global_log)))
        for island in self.islands:
            island.replace_worst(migrants)

# ----------------------------------------------------------------------
# Функция для запуска континента в процессе
# ----------------------------------------------------------------------

def run_continent(continent_id, trend_index, initial_log=None, queue=None):
    """Запускает континент в отдельном процессе."""
    print(f"Старт континента {continent_id}, тренд {trend_index}")
    cont = Continent(continent_id, trend_index, initial_log)
    cont.run_trend()
    best = cont.get_best_global()
    if queue:
        queue.put((continent_id, best[0], best[2]))  # отправляем лучшего
    # Сохраняем лог континента
    cont._save_log()
    print(f"Континент {continent_id} завершил тренд {trend_index}")
    return best

# ----------------------------------------------------------------------
# Загрузка/сохранение глобального лога
# ----------------------------------------------------------------------

def load_global_log():
    if os.path.exists(GLOBAL_LOG):
        with open(GLOBAL_LOG, 'r') as f:
            return json.load(f)
    return []

def save_global_log(log):
    with open(GLOBAL_LOG, 'w') as f:
        json.dump(log, f)

def update_global_log(new_entries):
    """Добавляет новые записи (список (fitness, genotype)) в глобальный лог."""
    log = load_global_log()
    log.extend(new_entries)
    log.sort(reverse=True, key=lambda x: x[0])
    log = log[:GLOBAL_LOG_SIZE]
    save_global_log(log)

# ----------------------------------------------------------------------
# Битва континентов
# ----------------------------------------------------------------------

def battle_continents(continents_data):
    """
    continents_data: список словарей с информацией о континентах
    Проводит битву: худшие острова удаляются, лучшие получают буст.
    Возвращает обновлённые данные для новых континентов.
    """
    # Собираем все острова со всех континентов с их лучшими фитнесами
    all_islands = []
    for cont in continents_data:
        for island_data in cont['islands']:
            all_islands.append({
                'continent_id': cont['id'],
                'island_idx': island_data['idx'],
                'best_fitness': island_data['best_fitness'],
                'population': island_data['population'],
                'genotypes': island_data['genotypes']
            })

    # Сортируем по лучшему фитнесу
    all_islands.sort(key=lambda x: x['best_fitness'], reverse=True)

    # Определяем, сколько островов оставить (BATTLE_ELITE_RATIO от общего числа)
    num_elite = int(len(all_islands) * BATTLE_ELITE_RATIO)
    elite_islands = all_islands[:num_elite]

    # Остальные удаляются (их данные не сохраняются)

    # Усиливаем элитные острова: увеличиваем популяцию и поколения
    for island in elite_islands:
        island['population'] = int(island['population'] * BATTLE_POP_BOOST)
        island['generations'] = int(island['generations'] * BATTLE_GEN_BOOST)

    # Перераспределяем элитные острова по континентам (равномерно)
    new_continents = []
    cont_size = len(elite_islands) // CONTINENTS
    for c in range(CONTINENTS):
        start = c * cont_size
        end = (c+1) * cont_size if c < CONTINENTS-1 else len(elite_islands)
        cont_islands = elite_islands[start:end]
        new_continents.append({
            'id': c,
            'islands': cont_islands
        })

    return new_continents

# ----------------------------------------------------------------------
# Основной координатор
# ----------------------------------------------------------------------

def main():
    print("🚀 Запуск многоконтинентальной эволюции")
    print(f"Континентов: {CONTINENTS}, островов на континенте: {ISLANDS_PER_CONTINENT}")
    print("Тренды:", TRENDS)

    global_log = load_global_log()
    all_best = []  # для финального лога

    # Проходим по всем трендам
    for trend_idx, (pop, gens) in enumerate(TRENDS):
        print(f"\n=== Тренд {trend_idx}: популяция {pop}, поколений {gens} ===")

        # Запускаем континенты параллельно
        processes = []
        queue = mp.Queue()
        for cid in range(CONTINENTS):
            # Каждому континенту даём копию глобального лога (для инъекции)
            p = mp.Process(target=run_continent, args=(cid, trend_idx, global_log, queue))
            processes.append(p)
            p.start()

        # Собираем результаты
        for p in processes:
            p.join()

        # Собираем лучших с каждого континента
        best_list = []
        while not queue.empty():
            cid, genotype, fitness_val = queue.get()
            best_list.append((fitness_val, genotype))
            all_best.append((fitness_val, genotype))

        # Обновляем глобальный лог
        update_global_log(best_list)
        global_log = load_global_log()

        print(f"Тренд {trend_idx} завершён. Лучшие фитнесы: {[f for f,_ in best_list]}")

    print("\n=== Все тренды завершены, начинается битва континентов ===")

    # Здесь нужно собрать данные о всех островах со всех континентов за последний тренд
    # Для простоты предположим, что у нас есть файлы логов континентов
    continents_data = []
    for cid in range(CONTINENTS):
        filename = f"continent_{cid}_log.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                cont_log = json.load(f)
            # Преобразуем в формат для битвы
            islands_data = []
            for idx in range(ISLANDS_PER_CONTINENT):
                # Здесь нужно реальные данные островов, но у нас только логи.
                # В качестве заглушки используем лучшие фитнесы из лога.
                # В реальности нужно сохранять состояния островов.
                best_fit = cont_log[0][0] if cont_log else 0
                islands_data.append({
                    'idx': idx,
                    'best_fitness': best_fit,
                    'population': TRENDS[-1][0],  # последняя популяция
                    'generations': TRENDS[-1][1],  # последние поколения
                    'genotypes': []  # здесь должны быть генотипы острова
                })
            continents_data.append({
                'id': cid,
                'islands': islands_data
            })

    # Проводим битву
    new_continents = battle_continents(continents_data)

    print("Битва завершена. Новое распределение островов по континентам:")
    for cont in new_continents:
        print(f"Континент {cont['id']}: островов {len(cont['islands'])}")

    # В новой версии нужно продолжить эволюцию с усиленными островами,
    # но для демонстрации мы просто сохраним финальный лог.

    # Сохраняем финальный лог (100 лучших из all_best)
    all_best.sort(reverse=True, key=lambda x: x[0])
    final_best = all_best[:FINAL_LOG_SIZE]
    with open(FINAL_LOG, 'w') as f:
        json.dump(final_best, f)

    print(f"\n✅ Финальный лог сохранён в {FINAL_LOG} с {len(final_best)} записями.")
    print("Лучший фитнес за всё время:", final_best[0][0] if final_best else 0)

if __name__ == "__main__":
    main()