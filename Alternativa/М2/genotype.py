"""
genotype.py

Модуль для работы с генотипами в эволюционной системе нейросетей.
Предоставляет функции генерации, мутации, кроссинговера и валидации генотипов,
а также вспомогательные функции для анализа структуры.

Генотип — список целых чисел переменной длины, кодирующий архитектуру сети.
Формат:
    [num_input, 
     блок_нейрона_1, блок_нейрона_2, ..., блок_нейрона_N]
Где N = num_input + num_hidden + num_output (num_output пока фиксирован = 1).

Каждый блок нейрона:
    тип (0 = простой, 1 = адресный)
    если простой: затем 4 числа (два адреса: слой, локальный ID)
    если адресный: затем 16 чисел:
        - два фиксированных адреса (4 числа)
        - четыре адреса со смещением (8 чисел)
        - четыре действия для таблицы (4 числа)
"""

import random
from typing import List, Tuple, Optional

# Параметры (можно вынести в config, здесь для наглядности)
MAX_INPUT = 10                # максимум входных нейронов
MAX_HIDDEN = 50               # максимум скрытых (подбирается под лимит генов)
MAX_GENES = 1000              # максимальная длина генотипа
MAX_ADDRESSABLE = 100         # максимум адресных нейронов (включая выходной)
MIN_INPUT = 1                 # минимум входных
MIN_HIDDEN = 0                # минимум скрытых

# Вероятности для мутации
MUTATION_RATE = 0.1           # вероятность мутации одного гена
ADD_NEURON_PROB = 0.05        # вероятность добавить нейрон
DEL_NEURON_PROB = 0.05        # вероятность удалить нейрон
CHANGE_TYPE_PROB = 0.02       # вероятность смены типа нейрона (простой<->адресный)

# Диапазоны для адресов
LAYERS = [0, 1, 2]            # допустимые слои (0 - входной, 1 - скрытый, 2 - выходной)
DELTA_RANGE = (-2, 0)         # смещение по слоям (от -2 до 0, чтобы не выйти за пределы)

# ----------------------------------------------------------------------
# Вспомогательные функции для подсчёта
# ----------------------------------------------------------------------

def count_genes(genotype: List[int]) -> int:
    """Возвращает длину генотипа (число генов)."""
    return len(genotype)

def get_num_input(genotype: List[int]) -> int:
    """Извлекает число входных нейронов из генотипа."""
    return genotype[0] if genotype else 0

def get_num_neurons(genotype: List[int]) -> int:
    """Возвращает общее число нейронов (входные+скрытые+выходной)."""
    if not genotype:
        return 0
    # После первого гена идут блоки нейронов
    # Но чтобы точно определить число нейронов, нужно знать число входных и разобрать блоки.
    # Упростим: будем хранить num_input и num_hidden в генотипе? Но у нас только num_input в начале.
    # Значит, количество нейронов = num_input + (число блоков после первого) - num_input? Нет, число блоков = len(genotype)-1.
    # Но это общее число блоков, которое равно общему числу нейронов.
    return len(genotype) - 1

def get_num_addressable(genotype: List[int]) -> int:
    """Подсчитывает количество адресных нейронов в генотипе."""
    if len(genotype) < 2:
        return 0
    count = 0
    idx = 1
    while idx < len(genotype):
        ntype = genotype[idx]
        idx += 1
        if ntype == 0:  # простой
            idx += 4
        else:            # адресный
            count += 1
            idx += 16
    return count

def get_num_hidden(genotype: List[int]) -> int:
    """Возвращает количество скрытых нейронов (общее нейронов - входные - выходной)."""
    total = get_num_neurons(genotype)
    num_in = get_num_input(genotype)
    # Выходной всегда один
    return total - num_in - 1

def validate(genotype: List[int]) -> bool:
    """Проверяет, не превышены ли ограничения по генам и адресным нейронам."""
    if count_genes(genotype) > MAX_GENES:
        return False
    if get_num_addressable(genotype) > MAX_ADDRESSABLE:
        return False
    # Можно добавить проверку корректности адресов, но это уже в фитнесе
    return True

# ----------------------------------------------------------------------
# Генерация случайного генотипа
# ----------------------------------------------------------------------

def random_genotype() -> List[int]:
    """
    Генерирует случайный генотип, удовлетворяющий ограничениям.
    Стратегия: сначала выбираем num_input, затем генерируем скрытые нейроны,
    постепенно увеличивая их число, пока не упрёмся в лимит генов.
    """
    num_input = random.randint(MIN_INPUT, MAX_INPUT)
    # Выходной нейрон всегда один и будет адресным
    num_output = 1

    # Сначала грубо прикинем максимально возможное число скрытых, чтобы не перебирать
    # Оценка: на простой нейрон 5 генов, на адресный 17. Пусть половина адресных (грубо).
    # Но для простоты будем генерировать и проверять.
    max_hidden_estimate = (MAX_GENES - 1 - num_input*5) // 5  # если все простые
    max_hidden_estimate = min(max_hidden_estimate, MAX_HIDDEN)

    # Начнём с небольшого числа скрытых и будем увеличивать, пока не превысим лимит
    num_hidden = 0
    best_genotype = None
    for attempt in range(10):  # несколько попыток
        # Случайное число скрытых от 0 до max_hidden_estimate
        nh = random.randint(0, max_hidden_estimate)
        # Генерируем типы для всех нейронов
        types = [0] * (num_input + nh + num_output)
        # Выходной делаем адресным
        types[-1] = 1
        # Остальные: с вероятностью p делаем адресными, но не превышая MAX_ADDRESSABLE
        p_addr = min(0.3, MAX_ADDRESSABLE / (num_input + nh + num_output))
        addr_count = 1  # выходной уже адресный
        for i in range(len(types)-1):
            if addr_count >= MAX_ADDRESSABLE:
                types[i] = 0
            elif random.random() < p_addr:
                types[i] = 1
                addr_count += 1
            else:
                types[i] = 0

        # Теперь строим генотип
        genotype = [num_input]
        for i, typ in enumerate(types):
            genotype.append(typ)
            if typ == 0:
                # Простой нейрон: два адреса (слой, локальный)
                # Генерируем случайные адреса, они могут быть невалидными
                for _ in range(2):
                    layer = random.choice(LAYERS)
                    if layer == 1:
                        local = random.randint(0, max(0, nh-1))  # пока неизвестно точное число, приблизительно
                    else:
                        local = 0
                    genotype.extend([layer, local])
            else:
                # Адресный нейрон
                # fixed_targets (2 адреса)
                for _ in range(2):
                    layer = random.choice(LAYERS)
                    if layer == 1:
                        local = random.randint(0, max(0, nh-1))
                    else:
                        local = 0
                    genotype.extend([layer, local])
                # address_list (4 адреса со смещением)
                for _ in range(4):
                    delta = random.randint(*DELTA_RANGE)
                    # Целевой слой: для входного нейрона (слой 0) или выходного (слой 2) - но здесь мы не знаем,
                    # поэтому просто генерируем local_id, а при декодировании будет проверка.
                    # Для простоты local_id = 0 или случайный в пределах 0..nh-1
                    if delta + (0 if i < num_input else 2) == 1:  # если целевой слой скрытый
                        local = random.randint(0, max(0, nh-1))
                    else:
                        local = 0
                    genotype.extend([delta, local])
                # actions
                for _ in range(4):
                    genotype.append(random.randint(0, 4))

        # Проверяем лимиты
        if count_genes(genotype) <= MAX_GENES and get_num_addressable(genotype) <= MAX_ADDRESSABLE:
            # Успех
            best_genotype = genotype
            break
        # Иначе пробуем с другим num_hidden

    if best_genotype is None:
        # Если не получилось, генерируем минимальный
        genotype = [1, 1]  # 1 входной, выходной адресный
        # Добавим минимальные параметры для выходного (адресного)
        genotype.extend([0,0,0,0])  # фикс адреса (пока нули)
        for _ in range(4):
            genotype.extend([0,0])   # адреса со смещением
        for _ in range(4):
            genotype.append(0)       # действия
        best_genotype = genotype

    return best_genotype

# ----------------------------------------------------------------------
# Мутация
# ----------------------------------------------------------------------

def mutate(genotype: List[int]) -> List[int]:
    """
    Применяет мутации к генотипу с защитой выходного нейрона.
    """
    new = genotype[:]
    # Парсим структуру
    num_input = new[0]
    idx = 1
    blocks = []  # (start, end, type)
    while idx < len(new):
        start = idx
        ntype = new[idx]
        idx += 1
        if ntype == 0:
            end = idx + 4
            idx += 4
        else:
            end = idx + 16
            idx += 16
        blocks.append((start, end, ntype))

    total_neurons = len(blocks)
    num_hidden = total_neurons - num_input - 1
    output_block_idx = total_neurons - 1  # индекс последнего блока

    # 1. Точечные мутации
    for i in range(len(new)):
        if random.random() < MUTATION_RATE:
            if i == 0:
                new[i] = random.randint(MIN_INPUT, MAX_INPUT)
            else:
                # Ищем блок, содержащий i, и запоминаем его индекс
                for block_idx, (start, end, ntype) in enumerate(blocks):
                    if start <= i < end:
                        pos_in_block = i - start
                        if ntype == 0:
                            # Простой блок
                            if pos_in_block == 0:
                                # тип нейрона – не мутируем, если это выходной
                                if block_idx == output_block_idx:
                                    continue
                                # здесь можно добавить логику смены типа
                            elif pos_in_block in (1,3):  # layer
                                new[i] = random.choice(LAYERS)
                            elif pos_in_block in (2,4):  # local
                                layer = new[i-1] if pos_in_block == 2 else new[i-2]
                                if layer == 1:
                                    max_local = max(0, num_hidden-1)
                                    new[i] = random.randint(0, max_local) if max_local > 0 else 0
                                else:
                                    new[i] = 0
                        else:  # адресный блок
                            if pos_in_block == 0:
                                if block_idx == output_block_idx:
                                    continue  # не мутируем тип выходного
                            elif 1 <= pos_in_block <= 4:
                                if pos_in_block in (1,3):  # layer
                                    new[i] = random.choice(LAYERS)
                                else:  # local
                                    layer = new[i-1] if pos_in_block == 2 else new[i-2]
                                    if layer == 1:
                                        max_local = max(0, num_hidden-1)
                                        new[i] = random.randint(0, max_local) if max_local > 0 else 0
                                    else:
                                        new[i] = 0
                            elif 5 <= pos_in_block <= 12:
                                if pos_in_block in (5,7,9,11):  # delta
                                    new[i] = random.randint(*DELTA_RANGE)
                                else:  # local
                                    delta = new[i-1]
                                    target_layer = (0 if i < num_input else 2) + delta
                                    if target_layer == 1:
                                        max_local = max(0, num_hidden-1)
                                        new[i] = random.randint(0, max_local) if max_local > 0 else 0
                                    else:
                                        new[i] = 0
                            elif 13 <= pos_in_block <= 16:  # actions
                                new[i] = random.randint(0, 4)
                        break  # выходим из цикла поиска блока

    # 2. Добавление нейрона (оставляем как есть)
    if random.random() < ADD_NEURON_PROB:
        # ... ваш код добавления нейрона (вставляет перед выходным)
        pass

    # 3. Удаление нейрона (удаляем только скрытые)
    if num_hidden > 0 and random.random() < DEL_NEURON_PROB:
        hidden_blocks = blocks[num_input:num_input+num_hidden]
        if hidden_blocks:
            idx_to_remove = random.randrange(len(hidden_blocks))
            block_start, block_end, _ = hidden_blocks[idx_to_remove]
            new = new[:block_start] + new[block_end:]

    return new

# ----------------------------------------------------------------------
# Кроссинговер
# ----------------------------------------------------------------------

def crossover(g1: List[int], g2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Одноточечный кроссинговер на уровне генов.
    Точка разреза выбирается в пределах минимальной длины.
    """
    min_len = min(len(g1), len(g2))
    if min_len < 2:
        return g1[:], g2[:]
    point = random.randint(1, min_len-1)
    child1 = g1[:point] + g2[point:]
    child2 = g2[:point] + g1[point:]
    return child1, child2

# ----------------------------------------------------------------------
# Дополнительные функции для отладки
# ----------------------------------------------------------------------

def print_genotype_info(genotype: List[int]) -> None:
    """Выводит информацию о генотипе."""
    print(f"Длина генотипа: {count_genes(genotype)}")
    print(f"Входных нейронов: {get_num_input(genotype)}")
    print(f"Всего нейронов: {get_num_neurons(genotype)}")
    print(f"Скрытых нейронов: {get_num_hidden(genotype)}")
    print(f"Адресных нейронов: {get_num_addressable(genotype)}")
    print(f"Валидность: {validate(genotype)}")

# ----------------------------------------------------------------------
# Пример использования
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Генерация случайного генотипа:")
    g = random_genotype()
    print_genotype_info(g)
    print("Первые 50 генов:", g[:50])

    print("\nМутация:")
    g_mut = mutate(g)
    print_genotype_info(g_mut)

    print("\nКроссинговер двух копий:")
    c1, c2 = crossover(g, g_mut)
    print_genotype_info(c1)
    print_genotype_info(c2)