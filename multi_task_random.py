import random
import json
import time
from copy import deepcopy
from my_neuron import SimpleNeuron, AddressableNeuron
from network import Network
from genotype import random_genotype, build_network_from_genotype

def evaluate_bool_func(net, truth_table, max_steps=3):
    """
    truth_table: список из 4 ожидаемых выходов для комбинаций (0,0), (0,1), (1,0), (1,1)
    """
    total = len(net.neurons)
    # Проверка числа нейронов уже сделана до вызова
    output_idx = total - 1  # последний нейрон
    correct = 0
    start = time.perf_counter()
    for x in range(4):
        b0 = x & 1
        b1 = (x >> 1) & 1
        net.reset()
        net.external_input(0, b0)
        net.external_input(1, b1)
        steps = 0
        while not net.is_quiet() and steps < max_steps:
            net.step()
            steps += 1
        if net.neurons[output_idx].state == truth_table[x]:
            correct += 1
    end = time.perf_counter()
    acc = correct / 4
    if acc == 1.0:
        return acc, (end - start) * 1e6
    else:
        return acc, None

def evaluate_detector_101(net, max_steps=5):
    total = len(net.neurons)
    output_idx = total - 1
    # Тесты: каждая последовательность из 3 бит, ожидаемый выход после последнего бита
    tests = [
        ([0,0,0], 0),
        ([0,0,1], 0),
        ([0,1,0], 0),
        ([0,1,1], 0),
        ([1,0,0], 0),
        ([1,0,1], 1),   # после 101 должен быть 1
        ([1,1,0], 0),
        ([1,1,1], 0),
    ]
    correct = 0
    start = time.perf_counter()
    for seq, expected in tests:
        net.reset()
        for bit in seq:
            net.external_input(0, bit)
            steps = 0
            while not net.is_quiet() and steps < max_steps:
                net.step()
                steps += 1
        if net.neurons[output_idx].state == expected:
            correct += 1
    end = time.perf_counter()
    acc = correct / len(tests)
    if acc == 1.0:
        return acc, (end - start) * 1e6
    else:
        return acc, None


def evaluate_parity3(net, max_steps=5):
    total = len(net.neurons)
    output_idx = total - 1
    # Все 8 комбинаций
    tests = []
    for x in range(8):
        bits = [(x >> i) & 1 for i in range(3)]
        expected = bin(x).count('1') % 2
        tests.append((bits, expected))
    correct = 0
    start = time.perf_counter()
    for bits, expected in tests:
        net.reset()
        for i, b in enumerate(bits):
            net.external_input(i, b)
        steps = 0
        while not net.is_quiet() and steps < max_steps:
            net.step()
            steps += 1
        if net.neurons[output_idx].state == expected:
            correct += 1
    end = time.perf_counter()
    acc = correct / len(tests)
    if acc == 1.0:
        return acc, (end - start) * 1e6
    else:
        return acc, None


def evaluate_adder(net, max_steps=5):
    total = len(net.neurons)
    if total < 4:  # минимум 2 входа + 2 выхода
        return 0.0, None
    sum_idx = total - 2
    carry_idx = total - 1
    tests = [(0,0,0,0), (0,1,1,0), (1,0,1,0), (1,1,0,1)]
    correct = 0
    start = time.perf_counter()
    for a, b, exp_sum, exp_carry in tests:
        net.reset()
        net.external_input(0, a)
        net.external_input(1, b)
        steps = 0
        while not net.is_quiet() and steps < max_steps:
            net.step()
            steps += 1
        if net.neurons[sum_idx].state == exp_sum and net.neurons[carry_idx].state == exp_carry:
            correct += 1
    end = time.perf_counter()
    acc = correct / len(tests)
    if acc == 1.0:
        return acc, (end - start) * 1e6
    else:
        return acc, None




# --- Определение задач ---
tasks = [
    {
        'name': 'xor',
        'n_in': 2,
        'n_out': 1,
        'evaluate': lambda net: evaluate_bool_func(net, [0,1,1,0]),
        'max_steps': 3,
    },
    {
        'name': 'and',
        'n_in': 2,
        'n_out': 1,
        'evaluate': lambda net: evaluate_bool_func(net, [0,0,0,1]),
        'max_steps': 3,
    },
    {
        'name': 'or',
        'n_in': 2,
        'n_out': 1,
        'evaluate': lambda net: evaluate_bool_func(net, [0,1,1,1]),
        'max_steps': 3,
    },
    {
        'name': 'detector_101',
        'n_in': 1,
        'n_out': 1,
        'evaluate': evaluate_detector_101,
        'max_steps': 5,
    },
    {
        'name': 'parity3',
        'n_in': 3,
        'n_out': 1,
        'evaluate': evaluate_parity3,
        'max_steps': 5,
    },
    {
        'name': 'adder',
        'n_in': 2,
        'n_out': 2,
        'evaluate': evaluate_adder,
        'max_steps': 5,
    }
]

# Инициализация хранилища результатов
results = {}
for task in tasks:
    results[task['name']] = {
        'best_acc': 0.0,
        'best_acc_genotype': None,
        'best_time': float('inf'),
        'best_time_genotype': None
    }

# Параметры поиска
total_attempts = 100000
min_neurons = 3
max_neurons = 15

print(f"Запуск случайного поиска на {total_attempts} попыток")
print("Задачи:", [t['name'] for t in tasks])
print()

start_time_all = time.time()

for attempt in range(1, total_attempts + 1):
    # Генерация случайной сети
    num_neurons = random.randint(min_neurons, max_neurons)
    genotype = random_genotype(num_neurons)
    try:
        net = build_network_from_genotype(genotype, Network,
                                          {'simple': SimpleNeuron, 'addressable': AddressableNeuron})
    except Exception as e:
        # Если сеть не построилась, пропускаем
        continue

    # Тестируем на всех задачах
    for task in tasks:
        # Проверяем, хватает ли нейронов для размещения входов и выходов
        if task['n_in'] + task['n_out'] > num_neurons:
            continue  # не хватает нейронов, пропускаем задачу для этой сети

        # Вызов функции оценки
        acc, t_us = task['evaluate'](net)

        # Обновление статистики
        task_name = task['name']
        res = results[task_name]

        # Лучшая точность
        if acc > res['best_acc']:
            res['best_acc'] = acc
            res['best_acc_genotype'] = deepcopy(genotype)

        # Лучшее время при точности 1.0
        if acc == 1.0 and t_us is not None and t_us < res['best_time']:
            res['best_time'] = t_us
            res['best_time_genotype'] = deepcopy(genotype)

    # Прогресс
    if attempt % 1000 == 0:
        elapsed = time.time() - start_time_all
        print(f"Попытка {attempt}/{total_attempts} ({elapsed:.1f} с)")
        # Можно вывести промежуточные лучшие результаты
        for task_name, res in results.items():
            if res['best_acc'] == 1.0:
                print(f"  {task_name}: acc=1.0, time={res['best_time']:.3f} мкс")
            else:
                print(f"  {task_name}: best_acc={res['best_acc']:.4f}")

# Сохранение и финальный отчёт
print("\n" + "="*60)
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
print("="*60)

for task in tasks:
    name = task['name']
    res = results[name]

    # Сохраняем лучший по точности (если есть)
    if res['best_acc_genotype'] is not None:
        acc_filename = f"best_{name}_acc.json"
        with open(acc_filename, 'w') as f:
            json.dump(res['best_acc_genotype'], f, indent=2)
        print(f"{name:15} | точность {res['best_acc']:.4f} | сохранён в {acc_filename}")

    # Сохраняем лучший по времени (если есть точность 1.0)
    if res['best_time_genotype'] is not None:
        time_filename = f"best_{name}_time.json"
        with open(time_filename, 'w') as f:
            json.dump(res['best_time_genotype'], f, indent=2)
        print(f"{name:15} | точность 1.0 | время {res['best_time']:.3f} мкс | сохранён в {time_filename}")
    else:
        print(f"{name:15} | решение с точностью 1.0 не найдено")

total_time = time.time() - start_time_all
print(f"\nОбщее время выполнения: {total_time:.2f} с") 
