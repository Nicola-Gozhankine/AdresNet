"""
train_parity_adaptive.py

Адаптивный генетический алгоритм с динамическим увеличением ресурсов при застое.
Если за один запуск ГА (с фиксированными параметрами) не удалось улучшить лучшую точность,
размер популяции и число поколений удваиваются. При улучшении параметры сбрасываются к базовым.
"""

import random
from my_neuron import SimpleNeuron, AddressableNeuron, Network

# ------------------------------------------
# Базовые параметры (с них начинаем)
BASE_POP_SIZE = 100
BASE_GENERATIONS = 200
BASE_MUTATION_RATE = 0.1

# Диапазон числа скрытых нейронов
MIN_HIDDEN = 1
MAX_HIDDEN = 20

# Допустимые слои
VALID_LAYERS = [0, 1, 2]

# Для address_list: delta_layer может быть таким, чтобы целевой слой был 0,1,2
DELTA_LAYER_RANGE = (-2, 0)

# Для action_table: 0=ordinary, 1..4=адресный индекс
ACTION_VALUES = list(range(5))

# Штрафы
INVALID_ADDRESS_PENALTY = 0.1

# Целевая точность для остановки
TARGET_ACCURACY = 0.95

# Максимальные значения (чтобы не уйти в бесконечность)
MAX_POP_SIZE = 5000
MAX_GENERATIONS = 5000

# ------------------------------------------
def random_genotype(num_hidden_range=(MIN_HIDDEN, MAX_HIDDEN)):
    """Генерирует случайный генотип (33 гена) с учётом допустимого числа скрытых нейронов."""
    num_hidden = random.randint(*num_hidden_range)
    
    # Входной адресный нейрон (слой 0)
    in_fixed = []
    for _ in range(2):
        layer = random.choice(VALID_LAYERS)
        if layer == 1:
            local = random.randint(0, num_hidden-1)
        else:
            local = 0
        in_fixed.extend([layer, local])
    
    in_addr = []
    for _ in range(4):
        delta = random.randint(*DELTA_LAYER_RANGE)
        target_layer = 0 + delta
        if target_layer not in VALID_LAYERS:
            target_layer = random.choice(VALID_LAYERS)
        if target_layer == 1:
            local = random.randint(0, num_hidden-1)
        else:
            local = 0
        in_addr.extend([delta, local])
    
    in_actions = [random.choice(ACTION_VALUES) for _ in range(4)]
    
    # Выходной адресный нейрон (слой 2)
    out_fixed = []
    for _ in range(2):
        layer = random.choice(VALID_LAYERS)
        if layer == 1:
            local = random.randint(0, num_hidden-1)
        else:
            local = 0
        out_fixed.extend([layer, local])
    
    out_addr = []
    for _ in range(4):
        delta = random.randint(*DELTA_LAYER_RANGE)
        target_layer = 2 + delta
        if target_layer not in VALID_LAYERS:
            target_layer = random.choice(VALID_LAYERS)
        if target_layer == 1:
            local = random.randint(0, num_hidden-1)
        else:
            local = 0
        out_addr.extend([delta, local])
    
    out_actions = [random.choice(ACTION_VALUES) for _ in range(4)]
    
    return [num_hidden] + in_fixed + in_addr + in_actions + out_fixed + out_addr + out_actions

def genotype_to_parameters(genotype):
    num_hidden = genotype[0]
    idx = 1
    in_fixed = [(genotype[idx], genotype[idx+1]), (genotype[idx+2], genotype[idx+3])]; idx += 4
    in_addr = [(genotype[idx], genotype[idx+1]), (genotype[idx+2], genotype[idx+3]),
               (genotype[idx+4], genotype[idx+5]), (genotype[idx+6], genotype[idx+7])]; idx += 8
    in_actions = genotype[idx:idx+4]; idx += 4
    out_fixed = [(genotype[idx], genotype[idx+1]), (genotype[idx+2], genotype[idx+3])]; idx += 4
    out_addr = [(genotype[idx], genotype[idx+1]), (genotype[idx+2], genotype[idx+3]),
                (genotype[idx+4], genotype[idx+5]), (genotype[idx+6], genotype[idx+7])]; idx += 8
    out_actions = genotype[idx:idx+4]
    return num_hidden, in_fixed, in_addr, in_actions, out_fixed, out_addr, out_actions

def create_network_from_genotype(genotype):
    num_hidden, in_fixed, in_addr, in_actions, out_fixed, out_addr, out_actions = genotype_to_parameters(genotype)
    
    net = Network()
    
    def make_action_table(actions):
        keys = [(0,0), (0,1), (1,0), (1,1)]
        table = {}
        for i, key in enumerate(keys):
            val = actions[i]
            if val == 0:
                table[key] = 'ordinary'
            else:
                table[key] = ('address', val-1)
        return table
    
    # Входной адресный нейрон
    in_neuron = AddressableNeuron(layer=0,
                                   fixed_targets=in_fixed,
                                   address_list=in_addr,
                                   action_table=make_action_table(in_actions))
    in_id = net.add_neuron(in_neuron)
    
    # Скрытые нейроны (простые)
    hidden_ids = []
    for i in range(num_hidden):
        hidden = SimpleNeuron(layer=1, target0=(2,0), target1=(2,0))
        hidden_ids.append(net.add_neuron(hidden))
    
    # Выходной адресный нейрон
    out_neuron = AddressableNeuron(layer=2,
                                    fixed_targets=out_fixed,
                                    address_list=out_addr,
                                    action_table=make_action_table(out_actions))
    out_id = net.add_neuron(out_neuron)
    
    return net, in_id, out_id

def evaluate_network(net, in_id, out_id, sequence):
    outputs = []
    for bit in sequence:
        outgoing = net.step(external_inputs={in_id: bit})
        out_bit = 0
        for from_gid, to_gid, b in outgoing:
            if from_gid == out_id:
                out_bit = b
                break
        outputs.append(out_bit)
    return outputs

def fitness(genotype, num_sequences=10, seq_len=20):
    """Чистая точность минус штраф за невалидные адреса (без учёта сложности)."""
    num_hidden, in_fixed, in_addr, _, out_fixed, out_addr, _ = genotype_to_parameters(genotype)
    
    penalty = 0.0
    # Проверка адресов входного и выходного нейронов
    for (layer, local) in in_fixed + out_fixed:
        if layer not in VALID_LAYERS:
            penalty += INVALID_ADDRESS_PENALTY
        elif layer == 1 and (local < 0 or local >= num_hidden):
            penalty += INVALID_ADDRESS_PENALTY
        elif layer in (0,2) and local != 0:
            penalty += INVALID_ADDRESS_PENALTY
    
    # Можно добавить проверку address_list, но для простоты опустим
    
    total_correct = 0
    total_steps = 0
    for _ in range(num_sequences):
        seq = [random.randint(0,1) for _ in range(seq_len)]
        true_parity = []
        p = 0
        for bit in seq:
            p ^= bit
            true_parity.append(p)
        
        net, in_id, out_id = create_network_from_genotype(genotype)
        outputs = evaluate_network(net, in_id, out_id, seq)
        
        for out_bit, true_bit in zip(outputs, true_parity):
            if out_bit == true_bit:
                total_correct += 1
        total_steps += seq_len
    
    accuracy = total_correct / total_steps
    return max(0, accuracy - penalty)

def run_ga(pop_size, generations, mutation_rate, num_hidden_range=(MIN_HIDDEN, MAX_HIDDEN)):
    """Запускает ГА с заданными параметрами и возвращает лучшую особь и её точность."""
    population = [random_genotype(num_hidden_range) for _ in range(pop_size)]
    elite_size = max(1, pop_size // 10)
    tournament_size = max(2, pop_size // 20)
    
    best_overall = None
    best_accuracy = 0.0
    
    for gen in range(generations):
        fitnesses = [fitness(ind) for ind in population]
        
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        current_best = population[best_idx]
        current_acc = fitnesses[best_idx]
        
        if current_acc > best_accuracy:
            best_accuracy = current_acc
            best_overall = current_best
        
        # Отбор элит
        elites = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elite_inds = [population[i] for i in elites]
        
        new_population = elite_inds[:]
        
        # Турнирный отбор для остальных
        while len(new_population) < pop_size:
            # Выбор родителей
            parents = []
            for _ in range(2):
                best = None
                for _ in range(tournament_size):
                    idx = random.randrange(len(population))
                    if best is None or fitnesses[idx] > fitnesses[best]:
                        best = idx
                parents.append(population[best])
            
            # Кроссинговер
            point = random.randint(1, len(parents[0])-1)
            child = parents[0][:point] + parents[1][point:]
            
            # Мутация
            new_child = child[:]
            for i in range(len(new_child)):
                if random.random() < mutation_rate:
                    if i == 0:  # num_hidden
                        new_child[i] = random.randint(*num_hidden_range)
                    else:
                        # Упрощённая мутация: случайное значение из допустимого диапазона (можно улучшить)
                        if i in [1,3,17,19]:  # layer в fixed
                            new_child[i] = random.choice(VALID_LAYERS)
                        elif i in [5,7,9,11,21,23,25,27]:  # delta
                            new_child[i] = random.randint(*DELTA_LAYER_RANGE)
                        else:
                            # local или actions
                            new_child[i] = random.randint(0, 4)  # для local и actions подойдёт 0..4
            new_population.append(new_child)
        
        population = new_population[:pop_size]
        
        if gen % 50 == 0:
            print(f"  поколение {gen}, лучшая точность = {best_accuracy:.4f}")
    
    return best_overall, best_accuracy

# ------------------------------------------
if __name__ == "__main__":
    print("Адаптивный генетический алгоритм для задачи чётности")
    print("Целевая точность:", TARGET_ACCURACY)
    
    pop_size = BASE_POP_SIZE
    generations = BASE_GENERATIONS
    mutation_rate = BASE_MUTATION_RATE
    multiplier = 1
    best_acc_global = 0.0
    best_genotype_global = None
    attempt = 1
    
    while best_acc_global < TARGET_ACCURACY and pop_size <= MAX_POP_SIZE and generations <= MAX_GENERATIONS:
        print(f"\n--- Попытка {attempt} ---")
        print(f"Размер популяции: {pop_size}, поколений: {generations}, мутация: {mutation_rate}")
        
        genotype, acc = run_ga(pop_size, generations, mutation_rate)
        
        print(f"Результат попытки: точность = {acc:.4f}")
        
        if acc > best_acc_global:
            best_acc_global = acc
            best_genotype_global = genotype
            print(f"✅ Улучшение! Новый лучший результат: {best_acc_global:.4f}")
            # Сбрасываем множитель и параметры к базовым
            multiplier = 1
            pop_size = BASE_POP_SIZE
            generations = BASE_GENERATIONS
            mutation_rate = BASE_MUTATION_RATE
        else:
            print(f"❌ Улучшения нет. Увеличиваем ресурсы.")
            multiplier *= 2
            pop_size = min(BASE_POP_SIZE * multiplier, MAX_POP_SIZE)
            generations = min(BASE_GENERATIONS * multiplier, MAX_GENERATIONS)
            # mutation_rate можно тоже увеличить, но оставим пока без изменений
        
        attempt += 1
    
    print("\n=== Финальный результат ===")
    print(f"Лучшая достигнутая точность: {best_acc_global:.4f}")
    
    if best_genotype_global is not None:
        num_hidden, in_fixed, in_addr, in_actions, out_fixed, out_addr, out_actions = genotype_to_parameters(best_genotype_global)
        print(f"Скрытых нейронов: {num_hidden}")
        print("Входной нейрон:")
        print(f"  fixed_targets: {in_fixed}")
        print(f"  address_list: {in_addr}")
        print("Выходной нейрон:")
        print(f"  fixed_targets: {out_fixed}")
        print(f"  address_list: {out_addr}")
    
    # Тест на длинной последовательности
    test_seq = [random.randint(0,1) for _ in range(50)]
    true_parity = []
    p = 0
    for bit in test_seq:
        p ^= bit
        true_parity.append(p)
    
    net, in_id, out_id = create_network_from_genotype(best_genotype_global)
    outputs = evaluate_network(net, in_id, out_id, test_seq)
    correct = sum(1 for o, t in zip(outputs, true_parity) if o == t)
    print(f"На тестовой последовательности длиной 50: правильно {correct} из 50 ({correct/50:.2%})")