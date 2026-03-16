"""
train_island_multitask.py

Островная модель генетического алгоритма для мультизадачного обучения.
Три задачи: Delayed XOR, Counter mod 3, Detect '01'.
Несколько островов (популяций) эволюционируют независимо, периодически обмениваясь лучшими особями.
"""

import random
import math
from my_neuron import SimpleNeuron, AddressableNeuron, Network

# ------------------------------------------
# Параметры островов
NUM_ISLANDS = 4
ISLAND_POP_SIZE = 150
GENERATIONS_PER_ISLAND = 300
MIGRATION_INTERVAL = 20          # миграция каждые N поколений
MIGRANTS_PER_ISLAND = 5          # сколько лучших особей отправляем

# Параметры ГА (общие для всех островов)
MUTATION_RATE = 0.15
ELITE_SIZE = 15
TOURNAMENT_SIZE = 7

# Диапазон числа скрытых нейронов
MIN_HIDDEN = 1
MAX_HIDDEN = 20

# Допустимые слои
VALID_LAYERS = [0, 1, 2]

# Для address_list: delta_layer может быть таким, чтобы целевой слой был 0,1,2
DELTA_LAYER_RANGE = (-2, 0)

# Для action_table: 0=ordinary, 1..4=адресный индекс
ACTION_VALUES = list(range(5))

# Штраф за невалидные адреса
INVALID_ADDRESS_PENALTY = 0.05

# Параметры оценки
NUM_SEQUENCES = 8                # количество случайных последовательностей для оценки
SEQ_LEN = 30                     # длина каждой последовательности

# ------------------------------------------
# Функции для вычисления истинных значений задач

def true_delayed_xor(seq):
    """Delayed XOR: out[t] = in[t] XOR in[t-1], out[0] = 0."""
    out = [0]
    for t in range(1, len(seq)):
        out.append(seq[t] ^ seq[t-1])
    return out

def true_counter_mod3(seq):
    """Counter mod 3: out[t] = 1 если (сумма единиц с начала) mod 3 == 0."""
    out = []
    s = 0
    for bit in seq:
        s = (s + bit) % 3
        out.append(1 if s == 0 else 0)
    return out

def true_detect_01(seq):
    """Detect '01': out[t] = 1 если seq[t-1]==0 and seq[t]==1, out[0]=0."""
    out = [0]
    for t in range(1, len(seq)):
        out.append(1 if (seq[t-1] == 0 and seq[t] == 1) else 0)
    return out

# ------------------------------------------
# Генерация генотипа (как в прошлой версии)
def random_genotype(num_hidden_range=(MIN_HIDDEN, MAX_HIDDEN)):
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

def fitness_multitask(genotype, num_sequences=NUM_SEQUENCES, seq_len=SEQ_LEN):
    """
    Фитнес = средняя точность по трём задачам (Delayed XOR, Counter mod 3, Detect '01').
    """
    num_hidden, in_fixed, in_addr, _, out_fixed, out_addr, _ = genotype_to_parameters(genotype)
    
    # Штраф за невалидные адреса (упрощённо)
    penalty = 0.0
    for (layer, local) in in_fixed + out_fixed:
        if layer not in VALID_LAYERS:
            penalty += INVALID_ADDRESS_PENALTY
        elif layer == 1 and (local < 0 or local >= num_hidden):
            penalty += INVALID_ADDRESS_PENALTY
        elif layer in (0,2) and local != 0:
            penalty += INVALID_ADDRESS_PENALTY
    
    total_correct = [0, 0, 0]   # для трёх задач
    total_steps = 0
    
    for _ in range(num_sequences):
        # Генерируем случайную последовательность
        seq = [random.randint(0,1) for _ in range(seq_len)]
        
        # Истинные значения для каждой задачи
        true_xor = true_delayed_xor(seq)
        true_mod3 = true_counter_mod3(seq)
        true_01 = true_detect_01(seq)
        
        # Прогоняем сеть
        net, in_id, out_id = create_network_from_genotype(genotype)
        outputs = evaluate_network(net, in_id, out_id, seq)
        
        # Сравниваем
        for t in range(seq_len):
            if outputs[t] == true_xor[t]:
                total_correct[0] += 1
            if outputs[t] == true_mod3[t]:
                total_correct[1] += 1
            if outputs[t] == true_01[t]:
                total_correct[2] += 1
        total_steps += seq_len
    
    # Точность по каждой задаче
    acc_xor = total_correct[0] / total_steps
    acc_mod3 = total_correct[1] / total_steps
    acc_01 = total_correct[2] / total_steps
    
    # Средняя точность
    avg_acc = (acc_xor + acc_mod3 + acc_01) / 3.0
    
    # Итоговый фитнес с учётом штрафа
    return max(0, avg_acc - penalty)

def mutate(genotype):
    """Мутация с учётом допустимых диапазонов."""
    new = genotype[:]
    num_hidden = new[0]
    for i in range(len(new)):
        if random.random() < MUTATION_RATE:
            if i == 0:
                new[i] = random.randint(MIN_HIDDEN, MAX_HIDDEN)
                num_hidden = new[i]
            else:
                # Определяем тип гена по индексу (упрощённо)
                if 1 <= i <= 4:  # in_fixed
                    if i % 2 == 1:  # layer
                        new[i] = random.choice(VALID_LAYERS)
                    else:  # local
                        layer = new[i-1]
                        if layer == 1:
                            new[i] = random.randint(0, num_hidden-1)
                        else:
                            new[i] = 0
                elif 5 <= i <= 12:  # in_addr
                    idx = i - 5
                    if idx % 2 == 0:  # delta
                        new[i] = random.randint(*DELTA_LAYER_RANGE)
                    else:  # local
                        delta = new[i-1]
                        target_layer = 0 + delta
                        if target_layer == 1:
                            new[i] = random.randint(0, num_hidden-1)
                        else:
                            new[i] = 0
                elif 13 <= i <= 16:  # in_actions
                    new[i] = random.choice(ACTION_VALUES)
                elif 17 <= i <= 20:  # out_fixed
                    if i % 2 == 1:
                        new[i] = random.choice(VALID_LAYERS)
                    else:
                        layer = new[i-1]
                        if layer == 1:
                            new[i] = random.randint(0, num_hidden-1)
                        else:
                            new[i] = 0
                elif 21 <= i <= 28:  # out_addr
                    idx = i - 21
                    if idx % 2 == 0:
                        new[i] = random.randint(*DELTA_LAYER_RANGE)
                    else:
                        delta = new[i-1]
                        target_layer = 2 + delta
                        if target_layer == 1:
                            new[i] = random.randint(0, num_hidden-1)
                        else:
                            new[i] = 0
                elif 29 <= i <= 32:  # out_actions
                    new[i] = random.choice(ACTION_VALUES)
    return new

def crossover(p1, p2):
    point = random.randint(1, len(p1)-1)
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
    return child1, child2

def selection(population, fitnesses):
    best = None
    for _ in range(TOURNAMENT_SIZE):
        idx = random.randrange(len(population))
        if best is None or fitnesses[idx] > fitnesses[best]:
            best = idx
    return population[best]

def evolve_island(population, generations):
    """Эволюция одной островной популяции в течение заданного числа поколений."""
    for gen in range(generations):
        fitnesses = [fitness_multitask(ind) for ind in population]
        
        # Элиты
        elites = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:ELITE_SIZE]
        elite_inds = [population[i] for i in elites]
        
        new_population = elite_inds[:]
        
        while len(new_population) < len(population):
            p1 = selection(population, fitnesses)
            p2 = selection(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_population.extend([c1, c2])
        
        population = new_population[:len(population)]
    
    # Возвращаем лучшую особь и её фитнес
    fitnesses = [fitness_multitask(ind) for ind in population]
    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    return population[best_idx], fitnesses[best_idx], population

def migrate(islands):
    """Обмен мигрантами между островами."""
    # Собираем лучших с каждого острова
    migrants_pool = []
    for island in islands:
        fitnesses = [fitness_multitask(ind) for ind in island]
        best_indices = sorted(range(len(island)), key=lambda i: fitnesses[i], reverse=True)[:MIGRANTS_PER_ISLAND]
        migrants_pool.extend([island[i] for i in best_indices])
    
    # Перемешиваем пул мигрантов
    random.shuffle(migrants_pool)
    
    # Каждому острову отправляем по одному мигранту (случайному из пула)
    for i in range(len(islands)):
        # Берём одного мигранта
        if not migrants_pool:
            break
        migrant = migrants_pool.pop()
        
        # Заменяем худшего на острове
        fitnesses = [fitness_multitask(ind) for ind in islands[i]]
        worst_idx = min(range(len(islands[i])), key=lambda j: fitnesses[j])
        islands[i][worst_idx] = migrant

# ------------------------------------------
if __name__ == "__main__":
    print("🏝️ Островная мультизадачная модель")
    print(f"Задачи: Delayed XOR, Counter mod 3, Detect '01'")
    print(f"Островов: {NUM_ISLANDS}, размер популяции на острове: {ISLAND_POP_SIZE}")
    print(f"Миграция каждые {MIGRATION_INTERVAL} поколений")
    
    # Инициализация островов
    islands = []
    for _ in range(NUM_ISLANDS):
        pop = [random_genotype() for _ in range(ISLAND_POP_SIZE)]
        islands.append(pop)
    
    best_global_fitness = 0.0
    best_global_genotype = None
    
    # Основной цикл
    for cycle in range(GENERATIONS_PER_ISLAND // MIGRATION_INTERVAL):
        print(f"\n--- Цикл {cycle+1} (поколения {cycle*MIGRATION_INTERVAL} - {(cycle+1)*MIGRATION_INTERVAL-1}) ---")
        
        # Эволюция на каждом острове
        for i in range(NUM_ISLANDS):
            print(f"  Остров {i+1}: эволюция...")
            best_ind, best_fit, islands[i] = evolve_island(islands[i], MIGRATION_INTERVAL)
            print(f"    Лучший фитнес на острове: {best_fit:.4f}")
            if best_fit > best_global_fitness:
                best_global_fitness = best_fit
                best_global_genotype = best_ind
                print(f"    🏆 Новый глобальный рекорд: {best_global_fitness:.4f}")
        
        # Миграция
        migrate(islands)
        print("  Миграция завершена")
    
    print("\n=== Финальный результат ===")
    print(f"Лучшая средняя точность: {best_global_fitness:.4f}")
    
    if best_global_genotype is not None:
        num_hidden, in_fixed, in_addr, in_actions, out_fixed, out_addr, out_actions = genotype_to_parameters(best_global_genotype)
        print(f"Скрытых нейронов: {num_hidden}")
        print("Входной нейрон:")
        print(f"  fixed_targets: {in_fixed}")
        print(f"  address_list: {in_addr}")
        print("Выходной нейрон:")
        print(f"  fixed_targets: {out_fixed}")
        print(f"  address_list: {out_addr}")
        
        # Тест на длинной последовательности
        test_seq = [random.randint(0,1) for _ in range(50)]
        true_xor = true_delayed_xor(test_seq)
        true_mod3 = true_counter_mod3(test_seq)
        true_01 = true_detect_01(test_seq)
        
        net, in_id, out_id = create_network_from_genotype(best_global_genotype)
        outputs = evaluate_network(net, in_id, out_id, test_seq)
        
        correct_xor = sum(1 for o, t in zip(outputs, true_xor) if o == t)
        correct_mod3 = sum(1 for o, t in zip(outputs, true_mod3) if o == t)
        correct_01 = sum(1 for o, t in zip(outputs, true_01) if o == t)
        L = len(test_seq)
        print(f"\nТест на последовательности длиной {L}:")
        print(f"  Delayed XOR: правильно {correct_xor} из {L} ({correct_xor/L:.2%})")
        print(f"  Counter mod 3: правильно {correct_mod3} из {L} ({correct_mod3/L:.2%})")
        print(f"  Detect '01': правильно {correct_01} из {L} ({correct_01/L:.2%})")
        print(f"  Среднее: {(correct_xor+correct_mod3+correct_01)/(3*L):.2%}")