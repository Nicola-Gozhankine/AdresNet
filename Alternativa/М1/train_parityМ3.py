"""
train_parity_dynamic.py

Генетический алгоритм с динамической архитектурой.
- Количество нейронов в скрытом слое эволюционирует (от 1 до 5).
- При достижении точности 90% включается штраф за сложность (число нейронов).
- После этого запускается вторая фаза (до 500 поколений) для поиска минимальной архитектуры.
"""

import random
from my_neuron import SimpleNeuron, AddressableNeuron, Network

# ------------------------------------------
# Параметры генетического алгоритма
POP_SIZE = 150
GENERATIONS = 2000          # максимум поколений для первой фазы
SECOND_PHASE_GENS = 500     # сколько поколений длится вторая фаза
MUTATION_RATE = 0.1
ELITE_SIZE = 10
TOURNAMENT_SIZE = 5

# Диапазон числа скрытых нейронов
MIN_HIDDEN = 1
MAX_HIDDEN = 5

# Допустимые слои (фиксированы)
VALID_LAYERS = [0, 1, 2]

# Для каждого слоя определим возможные локальные ID позже, в зависимости от числа скрытых нейронов
# Выходной слой (2) всегда имеет один нейрон (local_id=0)
# Входной слой (0) всегда один нейрон (local_id=0)
# Слой 1 (скрытый) имеет переменное число нейронов

# Для address_list: delta_layer может быть таким, чтобы целевой слой был 0,1,2
DELTA_LAYER_RANGE = (-2, 0)   # от -2 до 0 (чтобы 2+delta = 0..2)

# Для action_table: 0=ordinary, 1..4=адресный индекс
ACTION_VALUES = list(range(5))

# Порог для перехода ко второй фазе
ACCURACY_THRESHOLD = 0.9
STABILITY_WINDOW = 10        # сколько поколений держится точность выше порога для переключения

# Коэффициент штрафа за сложность во второй фазе
COMPLEXITY_PENALTY = 0.02    # вычитается за каждый скрытый нейрон

# ------------------------------------------
def random_genotype():
    """
    Генотип: [num_hidden] + 16 чисел (fixed_targets (4) + address_list (8) + actions (4))
    num_hidden от MIN_HIDDEN до MAX_HIDDEN.
    """
    num_hidden = random.randint(MIN_HIDDEN, MAX_HIDDEN)
    
    # fixed_targets: два адреса (layer, local_id) с учётом текущего num_hidden
    fixed = []
    for _ in range(2):
        layer = random.choice(VALID_LAYERS)
        if layer == 1:
            local = random.randint(0, num_hidden-1)
        else:
            local = 0
        fixed.extend([layer, local])
    
    # address_list: 4 адреса (delta_layer, local_id) с учётом, что целевой слой должен существовать
    addr = []
    for _ in range(4):
        delta = random.randint(*DELTA_LAYER_RANGE)
        target_layer = 2 + delta
        if target_layer == 1:
            local = random.randint(0, num_hidden-1)
        else:
            local = 0
        addr.extend([delta, local])
    
    # actions
    actions = [random.choice(ACTION_VALUES) for _ in range(4)]
    
    return [num_hidden] + fixed + addr + actions

def genotype_to_parameters(genotype):
    """Разбирает генотип на части: num_hidden, fixed, addr, actions."""
    num_hidden = genotype[0]
    fixed = [(genotype[1], genotype[2]), (genotype[3], genotype[4])]
    addr = [(genotype[5], genotype[6]), (genotype[7], genotype[8]),
            (genotype[9], genotype[10]), (genotype[11], genotype[12])]
    actions = genotype[13:17]
    return num_hidden, fixed, addr, actions

def create_network_from_genotype(genotype):
    """Создаёт сеть с заданным числом скрытых нейронов."""
    num_hidden, fixed, addr, actions = genotype_to_parameters(genotype)
    
    net = Network()
    
    # Входной слой (0): один простой нейрон
    # Его адресаты: на все скрытые нейроны (равномерно? для простоты пусть target0 на первый, target1 на последний)
    # Но у простого нейрона только два адресата. Если скрытых больше двух, нужно выбрать два.
    # Для простоты сделаем так: target0 = (1,0), target1 = (1, min(1, num_hidden-1)) — то есть либо оба на первый, либо на первый и второй.
    # Можно эволюционировать и это, но для начала упростим.
    # Важно: входной нейрон не является предметом эволюции в данной версии.
    if num_hidden == 1:
        in_neuron = SimpleNeuron(layer=0, target0=(1,0), target1=(1,0))
    else:
        in_neuron = SimpleNeuron(layer=0, target0=(1,0), target1=(1,1))
    in_id = net.add_neuron(in_neuron)
    
    # Скрытый слой (1): num_hidden простых нейронов
    # Их адресаты: все ведут на выходной нейрон (слой 2, local_id=0)
    hidden_ids = []
    for i in range(num_hidden):
        hidden = SimpleNeuron(layer=1, target0=(2,0), target1=(2,0))
        hidden_ids.append(net.add_neuron(hidden))
    
    # Выходной слой (2): один адресный нейрон
    # Преобразуем actions в action_table
    action_table = {}
    keys = [(0,0), (0,1), (1,0), (1,1)]
    for i, key in enumerate(keys):
        val = actions[i]
        if val == 0:
            action_table[key] = 'ordinary'
        else:
            action_table[key] = ('address', val-1)
    
    out_neuron = AddressableNeuron(layer=2,
                                   fixed_targets=fixed,
                                   address_list=addr,
                                   action_table=action_table)
    out_id = net.add_neuron(out_neuron)
    
    return net, in_id, out_id

def evaluate_network(net, in_id, out_id, sequence):
    """Прогоняет сеть на последовательности, возвращает список выходов."""
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

def fitness(genotype, num_sequences=5, seq_len=20, phase=1):
    """
    phase=1: только точность
    phase=2: точность - penalty * num_hidden
    """
    num_hidden, fixed, addr, _ = genotype_to_parameters(genotype)
    
    # Проверка валидности адресов (штрафуем невалидные)
    penalty = 0.0
    # fixed_targets
    for (layer, local) in fixed:
        if layer not in VALID_LAYERS:
            penalty += 0.03
        elif layer == 1 and (local < 0 or local >= num_hidden):
            penalty += 0.03
        elif layer in (0,2) and local != 0:
            penalty += 0.03
    # address_list
    for (delta, local) in addr:
        target_layer = 2 + delta
        if target_layer not in VALID_LAYERS:
            penalty += 0.03
        elif target_layer == 1 and (local < 0 or local >= num_hidden):
            penalty += 0.03
        elif target_layer in (0,2) and local != 0:
            penalty += 0.03
    
    # Оценка точности на последовательностях
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
    base_fitness = max(0, accuracy - penalty)
    
    if phase == 1:
        return base_fitness
    else:
        # Во второй фазе штрафуем за большое число нейронов
        complexity = num_hidden  # можно добавить и другие метрики
        return base_fitness - COMPLEXITY_PENALTY * complexity

def selection(population, fitnesses):
    """Турнирный отбор."""
    best = None
    for _ in range(TOURNAMENT_SIZE):
        idx = random.randrange(len(population))
        if best is None or fitnesses[idx] > fitnesses[best]:
            best = idx
    return population[best]

def crossover(p1, p2):
    """Одноточечный кроссинговер для генотипов одинаковой длины (17)."""
    point = random.randint(1, len(p1)-1)
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
    return child1, child2

def mutate(genotype):
    """Мутация с учётом допустимых диапазонов для каждого гена."""
    new = genotype[:]
    num_hidden = new[0]
    
    for i in range(len(new)):
        if random.random() < MUTATION_RATE:
            if i == 0:  # num_hidden
                new[i] = random.randint(MIN_HIDDEN, MAX_HIDDEN)
                num_hidden = new[i]  # обновляем для последующих проверок
            elif 1 <= i <= 4:  # fixed_targets (layer или local)
                idx = i - 1
                if idx % 2 == 0:  # layer
                    new[i] = random.choice(VALID_LAYERS)
                else:              # local_id
                    layer = new[i-1]  # соответствующий слой (уже может быть изменён в этом цикле, но для простоты используем старое значение)
                    # Чтобы избежать проблем, лучше пересчитывать после цикла, но для мутации допустимо
                    if layer == 1:
                        new[i] = random.randint(0, num_hidden-1)
                    else:
                        new[i] = 0
            elif 5 <= i <= 12:  # address_list (delta или local)
                idx = i - 5
                if idx % 2 == 0:  # delta
                    new[i] = random.randint(*DELTA_LAYER_RANGE)
                else:              # local
                    # нужно знать целевую layer, которая зависит от delta
                    delta = new[i-1]  # берём предыдущий (delta) — он может быть изменён в этом же цикле, но используем как есть
                    target_layer = 2 + delta
                    if target_layer == 1:
                        new[i] = random.randint(0, num_hidden-1)
                    else:
                        new[i] = 0
            else:  # actions (13..16)
                new[i] = random.choice(ACTION_VALUES)
    return new

def print_network_details(genotype):
    """Выводит подробную информацию о сети."""
    num_hidden, fixed, addr, actions = genotype_to_parameters(genotype)
    print(f"Скрытых нейронов: {num_hidden}")
    print(f"fixed_targets: {fixed}")
    print(f"address_list: {addr}")
    print("action_table:")
    keys = [(0,0), (0,1), (1,0), (1,1)]
    for i, key in enumerate(keys):
        val = actions[i]
        if val == 0:
            print(f"  {key} -> ordinary")
        else:
            print(f"  {key} -> ('address', {val-1})")
    print()

def genetic_algorithm():
    population = [random_genotype() for _ in range(POP_SIZE)]
    phase = 1  # начинаем с первой фазы
    best_accuracy_history = []
    generations_done = 0
    stable_count = 0
    
    for gen in range(GENERATIONS + SECOND_PHASE_GENS):
        if gen >= GENERATIONS and phase == 1:
            # Если первая фаза не достигла порога за отведённые поколения, принудительно переключаем
            print(f"\nДостигнут лимит первой фазы ({GENERATIONS} поколений). Переход ко второй фазе.")
            phase = 2
        
        # Вычисляем фитнес в зависимости от фазы
        fitnesses = [fitness(ind, phase=phase) for ind in population]
        
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        best_fitness = fitnesses[best_idx]
        best_individual = population[best_idx]
        
        # Для отслеживания точности (без штрафа) используем отдельную оценку
        acc = fitness(best_individual, phase=1)  # чистая точность
        best_accuracy_history.append(acc)
        
        # Вывод информации
        num_hidden = best_individual[0]
        print(f"Поколение {gen} (фаза {phase}): лучший фитнес = {best_fitness:.4f}, точность = {acc:.4f}, скрытых = {num_hidden}")
        
        # Проверка перехода во вторую фазу
        if phase == 1 and acc >= ACCURACY_THRESHOLD:
            stable_count += 1
            if stable_count >= STABILITY_WINDOW:
                print(f"\nТочность {acc:.4f} держится {STABILITY_WINDOW} поколений. Переход ко второй фазе оптимизации.")
                phase = 2
                stable_count = 0
                # Сброим счётчик поколений для второй фазы? Нет, просто продолжаем.
        else:
            stable_count = 0
        
        # Элиты
        elites = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:ELITE_SIZE]
        elite_inds = [population[i] for i in elites]
        
        # Новое поколение
        new_population = elite_inds[:]
        while len(new_population) < POP_SIZE:
            p1 = selection(population, fitnesses)
            p2 = selection(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_population.extend([c1, c2])
        
        population = new_population[:POP_SIZE]
        generations_done = gen + 1
        
        # Если прошло уже GENERATIONS + SECOND_PHASE_GENS, выходим
        if gen >= GENERATIONS + SECOND_PHASE_GENS - 1:
            break
    
    # Финальная оценка лучшей особи
    fitnesses = [fitness(ind, phase=1) for ind in population]  # используем чистую точность
    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    best_genotype = population[best_idx]
    best_accuracy = fitnesses[best_idx]
    
    return best_genotype, best_accuracy

if __name__ == "__main__":
    print("Запуск генетического алгоритма с динамической архитектурой...")
    best_genotype, best_acc = genetic_algorithm()
    
    print("\n=== Результаты обучения ===")
    print(f"Лучшая точность: {best_acc:.4f}")
    print("Параметры лучшей сети:")
    print_network_details(best_genotype)
    
    # Тестирование на новой последовательности
    test_seq = [random.randint(0,1) for _ in range(30)]
    true_parity = []
    p = 0
    for bit in test_seq:
        p ^= bit
        true_parity.append(p)
    
    net, in_id, out_id = create_network_from_genotype(best_genotype)
    outputs = evaluate_network(net, in_id, out_id, test_seq)
    correct = sum(1 for o, t in zip(outputs, true_parity) if o == t)
    print(f"На тестовой последовательности длиной 30: правильно {correct} из 30 ({correct/30:.2%})")