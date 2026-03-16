#!/usr/bin/env python3
"""
Скрипт для пошаговой трассировки лучших сетей AdresNet из JSON-файлов.
Полностью соответствует логике оценки из multi_task_random.py.
"""

import json
import sys
from my_neuron import SimpleNeuron, AddressableNeuron
from network import Network
from genotype import build_network_from_genotype

# --- Определение задач (все данные для тестирования) ---
TASKS = {
    'xor': {
        'name': 'XOR',
        'n_in': 2,
        'n_out': 1,
        'test_inputs': [(0,0), (0,1), (1,0), (1,1)],
        'expected': [0, 1, 1, 0],
        'max_steps': 3
    },
    'and': {
        'name': 'AND',
        'n_in': 2,
        'n_out': 1,
        'test_inputs': [(0,0), (0,1), (1,0), (1,1)],
        'expected': [0, 0, 0, 1],
        'max_steps': 3
    },
    'or': {
        'name': 'OR',
        'n_in': 2,
        'n_out': 1,
        'test_inputs': [(0,0), (0,1), (1,0), (1,1)],
        'expected': [0, 1, 1, 1],
        'max_steps': 3
    },
    'detector_101': {
        'name': 'Detector 101',
        'n_in': 1,
        'n_out': 1,
        'test_inputs': [
            [0,0,0], [0,0,1], [0,1,0], [0,1,1],
            [1,0,0], [1,0,1], [1,1,0], [1,1,1]
        ],
        'expected': [0, 0, 0, 0, 0, 1, 0, 0],
        'max_steps': 5
    },
    'parity3': {
        'name': 'Parity-3',
        'n_in': 3,
        'n_out': 1,
        'test_inputs': [
            (0,0,0), (0,0,1), (0,1,0), (0,1,1),
            (1,0,0), (1,0,1), (1,1,0), (1,1,1)
        ],
        'expected': [0, 1, 1, 0, 1, 0, 0, 1],
        'max_steps': 5
    },
    'adder': {
        'name': 'Adder',
        'n_in': 2,
        'n_out': 2,
        'test_inputs': [(0,0), (0,1), (1,0), (1,1)],
        'expected': [(0,0), (1,0), (1,0), (0,1)],  # (sum, carry)
        'max_steps': 5
    }
}

def trace_network_on_test(net, task_name, test_idx, inputs, expected, max_steps):
    """
    Выполняет трассировку для одного теста.
    Возвращает True, если результат совпал с ожидаемым.
    """
    print(f"\n{'='*60}")
    print(f"ТЕСТ {test_idx+1}: {TASKS[task_name]['name']}")
    print(f"Вход: {inputs}")
    print(f"Ожидаемый выход: {expected}")
    print(f"{'='*60}")

    # Сброс сети перед тестом – строго как в evaluate_*
    net.reset()
    print("\n[Инициализация]")
    print(f"  Состояния: {[n.state for n in net.neurons]}")
    print(f"  Буферы:    {[n.inbox for n in net.neurons]}")
    print(f"  Очередь:   {list(net._queue)}")

    # --- Подача входных сигналов ---
    if task_name == 'detector_101':
        # Для детектора – последовательная подача с обработкой после каждого бита
        for bit_idx, bit in enumerate(inputs):
            net.external_input(0, bit)
            print(f"\n[Вход {bit_idx+1}] Нейрон 0 получил {bit}")
            print(f"  Состояния до обработки: {[n.state for n in net.neurons]}")
            print(f"  Буферы до обработки:    {[n.inbox for n in net.neurons]}")
            print(f"  Очередь до обработки:   {list(net._queue)}")

            # Обработка шагов после данного бита
            step_inner = 0
            while not net.is_quiet() and step_inner < max_steps:
                step_inner += 1
                gid = list(net._queue)[0]
                print(f"\n  --- Внутренний шаг {step_inner} (обрабатывается нейрон {gid}) ---")
                print(f"    Состояния до: {[n.state for n in net.neurons]}")
                print(f"    Буферы до:    {[n.inbox for n in net.neurons]}")
                print(f"    Очередь до:   {list(net._queue)}")

                # Извлекаем нейрон
                gid = net._queue.popleft()
                net._in_queue.discard(gid)
                neuron = net.neurons[gid]

                s_old = neuron.state
                inbox_old = neuron.inbox
                y, target = neuron.step()

                print(f"    Нейрон {gid}:")
                print(f"      режим = {'OR' if neuron.mode==0 else 'XOR'}")
                print(f"      старое состояние = {s_old}, старый буфер = {inbox_old}")
                x = inbox_old if neuron.mode==0 else (inbox_old & 1)
                print(f"      входной бит x = {x}")
                print(f"      выход y = {y}")
                print(f"      новое состояние = {neuron.state}")
                print(f"      целевой нейрон = {target}")

                if target is not None and 0 <= target < len(net.neurons):
                    net.neurons[target].receive(y)
                    net._enqueue(target)
                    print(f"    -> Сигнал {y} отправлен нейрону {target}")

                print(f"    Состояния после: {[n.state for n in net.neurons]}")
                print(f"    Буферы после:    {[n.inbox for n in net.neurons]}")
                print(f"    Очередь после:   {list(net._queue)}")

            if step_inner == 0:
                print("  (нет активности)")
            else:
                print(f"  После обработки бита {bit_idx+1} сеть успокоилась за {step_inner} шагов.")

        # После всех битов может остаться активность – обработаем её с тем же ограничением
        step_extra = 0
        while not net.is_quiet() and step_extra < max_steps:
            step_extra += 1
            print(f"\n--- Дополнительный шаг после всех входов (обрабатывается нейрон {list(net._queue)[0]}) ---")
            net.step()  # для краткости без деталей, но можно добавить
        if step_extra > 0:
            print(f"  (потребовалось {step_extra} дополнительных шагов)")

    else:
        # Для всех остальных задач – подаём все входы сразу
        for i, bit in enumerate(inputs):
            net.external_input(i, bit)
            print(f"\n[Вход] Нейрон {i} получил {bit}")
        print(f"\nСостояния после входов: {[n.state for n in net.neurons]}")
        print(f"Буферы: {[n.inbox for n in net.neurons]}")
        print(f"Очередь: {list(net._queue)}")

        # Обработка шагов (точно как в evaluate_*)
        step = 0
        while not net.is_quiet() and step < max_steps:
            step += 1
            gid = list(net._queue)[0]
            print(f"\n--- Шаг {step} (обрабатывается нейрон {gid}) ---")
            print(f"  Состояния до: {[n.state for n in net.neurons]}")
            print(f"  Буферы до:    {[n.inbox for n in net.neurons]}")
            print(f"  Очередь до:   {list(net._queue)}")

            gid = net._queue.popleft()
            net._in_queue.discard(gid)
            neuron = net.neurons[gid]

            s_old = neuron.state
            inbox_old = neuron.inbox
            y, target = neuron.step()

            print(f"  Нейрон {gid}:")
            print(f"    режим = {'OR' if neuron.mode==0 else 'XOR'}")
            print(f"    старое состояние = {s_old}, старый буфер = {inbox_old}")
            x = inbox_old if neuron.mode==0 else (inbox_old & 1)
            print(f"    входной бит x = {x}")
            print(f"    выход y = {y}")
            print(f"    новое состояние = {neuron.state}")
            print(f"    целевой нейрон = {target}")

            if target is not None and 0 <= target < len(net.neurons):
                net.neurons[target].receive(y)
                net._enqueue(target)
                print(f"    -> Сигнал {y} отправлен нейрону {target}")

            print(f"  Состояния после: {[n.state for n in net.neurons]}")
            print(f"  Буферы после:    {[n.inbox for n in net.neurons]}")
            print(f"  Очередь после:   {list(net._queue)}")

        if step == 0:
            print("  (нет активности)")
        elif step == max_steps and not net.is_quiet():
            print(f"  (достигнут лимит шагов {max_steps}, очередь не пуста)")

    # --- Определение результата ---
    if task_name == 'adder':
        sum_idx = len(net.neurons) - 2
        carry_idx = len(net.neurons) - 1
        result = (net.neurons[sum_idx].state, net.neurons[carry_idx].state)
    else:
        out_idx = len(net.neurons) - 1
        result = net.neurons[out_idx].state

    success = (result == expected)
    print(f"\n>>> РЕЗУЛЬТАТ: {result} (ожидалось {expected}) -> {'✓' if success else '✗'}")
    return success

def trace_file(json_file, tasks_to_trace=None):
    """Загружает генотип из JSON и трассирует указанные задачи."""
    with open(json_file, 'r') as f:
        genotype = json.load(f)

    net = build_network_from_genotype(genotype, Network,
                                      {'simple': SimpleNeuron, 'addressable': AddressableNeuron})

    print(f"\nЗагружена сеть из {json_file}")
    print(f"Число нейронов: {len(net.neurons)}")
    print("Слои и локальные ID:")
    for gid, n in enumerate(net.neurons):
        ntype = 'simple' if hasattr(n, 'target_gids') else 'addressable'
        print(f"  {gid}: layer={n.layer}, local_id={n.local_id}, type={ntype}, mode={n.mode}")

    if tasks_to_trace is None:
        tasks_to_trace = TASKS.keys()

    for task_name in tasks_to_trace:
        if task_name not in TASKS:
            print(f"Неизвестная задача: {task_name}")
            continue
        task = TASKS[task_name]
        print(f"\n\n{'#'*60}")
        print(f"# ЗАДАЧА: {task['name']}")
        print(f"{'#'*60}")

        if task['n_in'] + task['n_out'] > len(net.neurons):
            print(f"Недостаточно нейронов (нужно минимум {task['n_in']+task['n_out']}, есть {len(net.neurons)}). Пропускаем.")
            continue

        correct = 0
        for idx, inputs in enumerate(task['test_inputs']):
            exp = task['expected'][idx]
            if trace_network_on_test(net, task_name, idx, inputs, exp, task['max_steps']):
                correct += 1

        print(f"\n>>> ИТОГО по задаче {task['name']}: {correct}/{len(task['test_inputs'])} правильных.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python trace_best.py <json_file> [задача1 задача2 ...]")
        print("Доступные задачи:", list(TASKS.keys()))
        sys.exit(1)

    json_file = sys.argv[1]
    tasks = sys.argv[2:] if len(sys.argv) > 2 else None
    trace_file(json_file, tasks)