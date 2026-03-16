# genotype_advanced.py
import random

# Константы для генов (максимальные значения, на практике будут меньше из-за штрафа сложности)
MAX_LAYER = 1000       # максимальный номер слоя (слои от 0 до MAX_LAYER-1)
MAX_LOCAL = 5000       # максимальный локальный ID в слое
ACTIONS = 5            # 0=ordinary, 1..4 = адресный индекс

# Размер блока одного нейрона: layer(1) + type(1) + mode(1) + fixed0(2) + fixed1(2) + addr(4*2) + actions(4) = 1+1+1+2+2+8+4 = 19
BLOCK_SIZE = 19

def random_genotype(num_neurons, max_layer=MAX_LAYER, max_local=MAX_LOCAL):
    # 1. Сначала генерируем только слои и типы (без адресов)
    base = []
    layers_list = []
    for _ in range(num_neurons):
        layer = random.randint(0, max_layer - 1)
        ntype = random.randint(0, 1)
        mode = random.randint(0, 1)
        base.append((layer, ntype, mode))
        layers_list.append(layer)
    
    # 2. Определяем уникальные слои и их размеры
    unique_layers = list(set(layers_list))
    layer_sizes = {layer: layers_list.count(layer) for layer in unique_layers}
    
    # 3. Генерируем полные блоки
    genotype = [num_neurons]
    for (layer, ntype, mode) in base:
        block = [layer, ntype, mode]
        
        # fixed0 – цель при y=0
        target_layer0 = random.choice(unique_layers)
        target_local0 = random.randint(0, layer_sizes[target_layer0] - 1)
        block.extend([target_layer0, target_local0])
        
        # fixed1 – цель при y=1
        target_layer1 = random.choice(unique_layers)
        target_local1 = random.randint(0, layer_sizes[target_layer1] - 1)
        block.extend([target_layer1, target_local1])
        
        # 4 адресных регистра (delta, local)
        for _ in range(4):
            # Генерируем delta, затем корректируем целевой слой
            delta = random.randint(-max_layer + 1, max_layer - 1)
            raw_target_layer = layer + delta
            # Если полученный слой не существует, выбираем случайный существующий
            if raw_target_layer not in unique_layers:
                target_layer = random.choice(unique_layers)
            else:
                target_layer = raw_target_layer
            local = random.randint(0, layer_sizes[target_layer] - 1)
            block.append(delta)   # сохраняем оригинальную дельту (для мутаций)
            block.append(local)
        
        # 4 действия
        for _ in range(4):
            block.append(random.randint(0, ACTIONS - 1))
        
        genotype.append(block)
    
    return genotype

#---------------------------------------------------------------------------------------------------------

# Вспомогательная функция для подсчёта размеров слоёв
def get_layer_counts(genotype):
    counts = {}
    for i in range(1, len(genotype)):
        layer = genotype[i][0]
        counts[layer] = counts.get(layer, 0) + 1
    return counts

def mutate(genotype, rate=0.1, max_layer=MAX_LAYER, max_local=MAX_LOCAL):
    """
    Мутирует генотип, сохраняя все адреса валидными.
    Слой нейрона не мутируется.
    """
    layer_counts = get_layer_counts(genotype)
    for i in range(1, len(genotype)):
        block = genotype[i]
        layer = block[0]  # слой не меняем

        for j in range(len(block)):
            if random.random() < rate:
                if j == 0:
                    continue
                elif j == 1:  # type
                    block[j] = random.randint(0, 1)
                elif j == 2:  # mode
                    block[j] = random.randint(0, 1)
                elif j in (3, 5):  # fixed layer
                    if layer_counts:
                        block[j] = random.choice(list(layer_counts.keys()))
                    else:
                        block[j] = 0
                elif j in (4, 6):  # fixed local
                    target_layer = block[j-1]
                    size = layer_counts.get(target_layer, 1)
                    block[j] = random.randint(0, size - 1)
                elif j in (7, 9, 11, 13):  # delta
                    if layer_counts:
                        target_layer = random.choice(list(layer_counts.keys()))
                        block[j] = target_layer - layer
                    else:
                        block[j] = 0
                elif j in (8, 10, 12, 14):  # addr local
                    delta = block[j-1]
                    target_layer = layer + delta
                    size = layer_counts.get(target_layer, 1)
                    block[j] = random.randint(0, size - 1)
                else:  # actions
                    block[j] = random.randint(0, ACTIONS - 1)
    return genotype

def crossover(g1, g2):
    """Одноточечный кроссовер с последующей коррекцией локальных ID."""
    blocks1 = g1[1:]
    blocks2 = g2[1:]
    n1 = len(blocks1)
    n2 = len(blocks2)
    if n1 == 0 or n2 == 0:
        return [g1[:], g2[:]]
    cut = random.randint(1, min(n1, n2))
    child1_blocks = blocks1[:cut] + blocks2[cut:]
    child2_blocks = blocks2[:cut] + blocks1[cut:]
    child1 = [len(child1_blocks)] + child1_blocks
    child2 = [len(child2_blocks)] + child2_blocks
    # Коррекция
    _fix_child(child1)
    _fix_child(child2)
    return child1, child2

def _fix_child(genotype):
    """Приводит локальные ID в соответствие с размерами слоёв."""
    layer_counts = get_layer_counts(genotype)
    for i in range(1, len(genotype)):
        block = genotype[i]
        layer = block[0]
        # fixed0 local
        target_layer = block[3]
        size = layer_counts.get(target_layer, 1)
        if block[4] >= size:
            block[4] = size - 1 if size > 0 else 0
        # fixed1 local
        target_layer = block[5]
        size = layer_counts.get(target_layer, 1)
        if block[6] >= size:
            block[6] = size - 1 if size > 0 else 0
        # addr locals
        for j in range(4):
            delta_idx = 7 + j*2
            local_idx = 8 + j*2
            target_layer = layer + block[delta_idx]
            size = layer_counts.get(target_layer, 1)
            if block[local_idx] >= size:
                block[local_idx] = size - 1 if size > 0 else 0
    return genotype




#---------------------------------------------------------------------------------------------------------
def decode(genotype):
    """
    Преобразует генотип в список параметров нейронов и карту слоёв.
    Возвращает:
        neurons: список словарей с полями layer, type, mode,
                 fixed0, fixed1 (кортежи (layer, local)),
                 addr (список из 4 кортежей (delta, local)),
                 actions (список из 4 int)
        layer_to_global: словарь {layer: [global_id, ...]}
    """
    num_neurons = genotype[0]
    neurons = []
    layer_to_global = {}
    global_id = 0
    for block in genotype[1:]:
        layer = block[0]
        neurons.append({
            'layer': layer,
            'type': block[1],
            'mode': block[2],
            'fixed0': (block[3], block[4]),
            'fixed1': (block[5], block[6]),
            'addr': [
                (block[7], block[8]),
                (block[9], block[10]),
                (block[11], block[12]),
                (block[13], block[14])
            ],
            'actions': block[15:19]
        })
        if layer not in layer_to_global:
            layer_to_global[layer] = []
        layer_to_global[layer].append(global_id)
        global_id += 1
    return neurons, layer_to_global

def fix_addresses(neurons, layer_to_global):
    """
    Корректирует адреса (fixed и address), чтобы локальные ID не выходили за пределы размера целевого слоя.
    Если целевой слой не существует, перенаправляет на слой 0, local 0.
    Изменяет словари neurons на месте.
    """
    # сначала узнаем размер каждого слоя
    layer_sizes = {layer: len(gids) for layer, gids in layer_to_global.items()}
    for neuron in neurons:
        # фиксированные цели
        for key in ['fixed0', 'fixed1']:
            layer, local = neuron[key]
            if layer not in layer_sizes:
                # слой не существует → перенаправляем на слой 0, local 0
                neuron[key] = (0, 0)
            else:
                size = layer_sizes[layer]
                if local >= size:
                    # берём по модулю или просто 0
                    neuron[key] = (layer, local % size if size > 0 else 0)
        # адресные регистры: каждый кортеж (delta, local) — но здесь local уже абсолютный, а delta потом применится
        # в процессе построения сети мы будем вычислять целевой слой как neuron['layer'] + delta,
        # поэтому local нужно корректировать относительно размера того слоя
        # Здесь мы не знаем delta, поэтому отложим коррекцию до построения сети.
        # Можно также сразу проверить, что local < MAX_LOCAL, но это уже заложено.
        pass
    return neurons

#---------------------------------------------------------------------------------------------------------

def build_network_from_genotype(genotype, network_class, neuron_classes):
    """
    Строит экземпляр сети по генотипу с коррекцией невалидных адресов.
    network_class: класс сети (должен иметь add_neuron, local_to_global и т.д.)
    neuron_classes: словарь {'simple': SimpleNeuron, 'addressable': AddressableNeuron}
    Возвращает сеть.
    """
    neurons_params, layer_to_global = decode(genotype)
    # Корректируем адреса (только fixed, адресные регистры требуют знания целевого слоя, что будет позже)
    fix_addresses(neurons_params, layer_to_global)

    net = network_class()
    # Добавляем все нейроны (пока без заполнения целей)
    for params in neurons_params:
        if params['type'] == 0:
            neuron = neuron_classes['simple'](layer=params['layer'], mode=params['mode'])
        else:
            neuron = neuron_classes['addressable'](layer=params['layer'], mode=params['mode'])
        net.add_neuron(neuron)

    # Теперь заполняем цели для каждого нейрона с учётом актуальных глобальных ID
    for gid, params in enumerate(neurons_params):
        neuron = net.neurons[gid]
        if params['type'] == 0:  # простой
            target0 = net.local_to_global(params['fixed0'][0], params['fixed0'][1])
            target1 = net.local_to_global(params['fixed1'][0], params['fixed1'][1])
            neuron.target_gids = [target0, target1]
        else:  # адресный
            # fixed цели
            fixed0 = net.local_to_global(params['fixed0'][0], params['fixed0'][1])
            fixed1 = net.local_to_global(params['fixed1'][0], params['fixed1'][1])
            neuron.fixed_gids = [fixed0, fixed1]

            # адресные регистры
            addr_gids = []
            for (delta, local) in params['addr']:
                target_layer = params['layer'] + delta
                if target_layer not in net.layer_to_global:
                    gid_target = net.local_to_global(0, 0)
                else:
                    size = len(net.layer_to_global[target_layer])
                    if local >= size:
                        local = local % size if size > 0 else 0
                    gid_target = net.local_to_global(target_layer, local)
                addr_gids.append(gid_target)
            neuron.address_gids = addr_gids

            # список действий (вместо action_table)
            action_list = []
            for val in params['actions']:
                if val == 0:
                    action_list.append('ordinary')
                else:
                    action_list.append(f'addr{val-1}')
            neuron.action_list = action_list

    return net

# Пример использования
if __name__ == "__main__":
    # создадим случайный генотип с 3 нейронами
    g = random_genotype(3)
    print("Генотип (первые 3 блока):", g[0], g[1][:5], '...')
    neurons, layer_map = decode(g)
    print("Раскодировано нейронов:", len(neurons))
    print("Слои и количество нейронов:", {layer: len(gids) for layer, gids in layer_map.items()})
    # мутация
    mutate(g, rate=0.3)
    # кроссовер
    g2 = random_genotype(4)
    c1, c2 = crossover(g, g2)
    print("Потомки имеют нейронов:", c1[0], c2[0])