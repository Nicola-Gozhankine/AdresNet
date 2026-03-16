"""
test_best_model_detailed.py

Загружает лучшую модель из best_model.json и проводит детальную трассировку
на выбранной задаче (parity или 101). Выводит потактово:
- вход, ожидаемый выход, фактический выход
- состояния всех нейронов
- все отправленные сигналы (откуда -> куда, бит)
- пояснение, что произошло в такте
"""


import json
import sys
from network_builder import genotype_to_network
from fitness import true_parity
from fitness_pattern101 import true_pattern_101

def trace_network_detailed(genotype, task="101", seq=None):
    if seq is None:
        seq = [1,0,1,1,0,1,0,0,1,0]  # тестовая последовательность по умолчанию

    # Выбираем функцию истинных значений
    if task == "parity":
        true_func = true_parity
        task_name = "чётность"
    elif task == "101":
        true_func = true_pattern_101
        task_name = "детектор '101'"
    else:
        raise ValueError("Неизвестная задача")

    try:
        net, input_ids, output_id = genotype_to_network(genotype)
    except Exception as e:
        print(f"Ошибка при создании сети: {e}")
        return

    print(f"\n=== Детальная трассировка для задачи {task_name} ===")
    print(f"Входная последовательность: {seq}")
    print("Такт | Вход | Ожид | Выход | Состояния (слой.локальный:сост) | Отправленные сигналы")
    print("-" * 80)

    net.reset_states()
    true_vals = true_func(seq)

    for t, bit in enumerate(seq):
        external = {gid: bit for gid in input_ids}
        outgoing = net.step(external_inputs=external)

        # Выходной бит сети
        out_bit = 0
        for from_gid, to_gid, b in outgoing:
            if from_gid == output_id:
                out_bit = b
                break

        # Состояния нейронов
        states = []
        for gid, neuron in enumerate(net.neurons):
            states.append(f"{neuron.layer}.{neuron.local_id}:{neuron.state}")

        # Отформатированные сигналы
        signals = []
        for from_gid, to_gid, b in outgoing:
            from_neuron = net.neurons[from_gid]
            to_neuron = net.neurons[to_gid] if to_gid is not None else None
            signals.append(f"{from_neuron.layer}.{from_neuron.local_id}->{to_neuron.layer}.{to_neuron.local_id}({b})")

        print(f"{t:4} | {bit:4} | {true_vals[t]:4} | {out_bit:5} | {', '.join(states):30} | {', '.join(signals)}")

        # Пояснение (опционально)
        if out_bit != true_vals[t]:
            print(f"      ❌ Ошибка: ожидалось {true_vals[t]}, получено {out_bit}")
        else:
            print(f"      ✅ Верно")

    print("=" * 80)

def main():
    # Загружаем модель
    with open("best_model.json", "r") as f:
        data = json.load(f)
    genotype = data["genotype"]
    print(f"Загружена модель с raw fitness = {data['raw']}")

    # Выбираем задачу
    task = input("Выберите задачу (parity / 101) [101]: ").strip() or "101"
    trace_network_detailed(genotype, task)

if __name__ == "__main__":
    main()