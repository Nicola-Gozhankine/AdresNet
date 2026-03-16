#!/usr/bin/env python3
import json
import time
import random
from my_neuron import SimpleNeuron, AddressableNeuron
from network import Network
from genotype import build_network_from_genotype

# Импортируем функции оценки из multi_task_random (или продублируем их здесь)
from multi_task_random import (
    evaluate_bool_func, evaluate_detector_101,
    evaluate_parity3, evaluate_adder
)

# Соответствие между именем задачи и файлом с лучшим генотипом
best_files = {
    'xor': 'best_xor_time.json',
    'and': 'best_and_time.json',
    'or': 'best_or_time.json',
    'detector_101': 'best_detector_101_time.json',
    'parity3': 'best_parity3_time.json',
    'adder': 'best_adder_time.json',
}

# Параметры задач (повторяем из multi_task_random)
tasks = {
    'xor': {
        'n_in': 2, 'n_out': 1,
        'evaluate': lambda net: evaluate_bool_func(net, [0,1,1,0]),
        'max_steps': 3
    },
    'and': {
        'n_in': 2, 'n_out': 1,
        'evaluate': lambda net: evaluate_bool_func(net, [0,0,0,1]),
        'max_steps': 3
    },
    'or': {
        'n_in': 2, 'n_out': 1,
        'evaluate': lambda net: evaluate_bool_func(net, [0,1,1,1]),
        'max_steps': 3
    },
    'detector_101': {
        'n_in': 1, 'n_out': 1,
        'evaluate': evaluate_detector_101,
        'max_steps': 5
    },
    'parity3': {
        'n_in': 3, 'n_out': 1,
        'evaluate': evaluate_parity3,
        'max_steps': 5
    },
    'adder': {
        'n_in': 2, 'n_out': 2,
        'evaluate': evaluate_adder,
        'max_steps': 5
    }
}

def benchmark_task(task_name, num_runs=10000):
    """Загружает лучший генотип для задачи и прогоняет его num_runs раз."""
    # Загрузка генотипа
    filename = best_files[task_name]
    with open(filename, 'r') as f:
        genotype = json.load(f)

    # Построение сети
    net = build_network_from_genotype(
        genotype, Network,
        {'simple': SimpleNeuron, 'addressable': AddressableNeuron}
    )

    # Получаем функцию оценки для этой задачи
    task_info = tasks[task_name]
    evaluate = task_info['evaluate']
    max_steps = task_info['max_steps']

    # Прогон
    correct_total = 0
    start_time = time.perf_counter()

    for _ in range(num_runs):
        # evaluate возвращает (accuracy, time_us) — нам нужна только точность
        acc, _ = evaluate(net)
        # Если acc == 1.0, значит все тесты в этом прогоне пройдены
        if acc == 1.0:
            correct_total += 1
        else:
            # Если вдруг ошибка, можно сразу вывести и остановиться
            print(f"Ошибка на прогоне {_} для задачи {task_name}!")
            # Здесь можно сохранить состояние сети для анализа
            break

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time_us = (total_time / num_runs) * 1e6

    print(f"\n=== {task_name} ===")
    print(f"Прогонов: {num_runs}")
    print(f"Успешных: {correct_total} / {num_runs}  (точность {correct_total/num_runs*100:.6f}%)")
    print(f"Общее время: {total_time:.3f} с")
    print(f"Среднее время на прогон: {avg_time_us:.3f} мкс")

if __name__ == "__main__":
    # Можно задать количество прогонов через аргумент командной строки
    import sys
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

    for task_name in best_files.keys():
        benchmark_task(task_name, num_runs) 