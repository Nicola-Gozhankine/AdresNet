"""
my_neuron.py

Реализация модели нейросети, предложенной пользователем.

Основные идеи:
- Нейроны имеют состояние (1 бит), которое переключается после каждого такта.
- Выходной бит зависит от состояния и входа: если состояние 0, выход = вход; если состояние 1, выход = инверсия входа.
- Есть два типа нейронов: простые (SimpleNeuron) и адресные (AddressableNeuron).
- Простой нейрон имеет два фиксированных адресата в следующем слое и отправляет сигнал одному из них в зависимости от выходного бита.
- Адресный нейрон также имеет два фиксированных адресата для обычного режима, а также список из 4 адресов (каждый задаётся смещением по слоям и локальным номером). Для каждой комбинации (состояние, вход) задано действие: либо обычный режим, либо отправка по одному из 4 адресов.
- Сеть работает синхронно по тактам. Сигналы накапливаются в inbox нейронов по ИЛИ.
"""

class Neuron:
    """Базовый класс для всех нейронов."""
    def __init__(self, layer, initial_state=0):
        """
        layer: номер слоя, к которому принадлежит нейрон (целое число).
        initial_state: начальное состояние (0 или 1).
        """
        self.layer = layer
        self.local_id = None          # будет установлен сетью при добавлении
        self.state = initial_state    # текущее состояние (0 или 1)
        self.inbox = None             # накопленные входные сигналы (0, 1 или None)

    def receive(self, bit):
        """
        Принять сигнал от другого нейрона.
        Сигналы объединяются по ИЛИ: если уже есть 1, остаётся 1.
        """
        if self.inbox is None:
            self.inbox = bit
        else:
            self.inbox |= bit

    def step(self):
        """
        Выполнить один такт работы нейрона.
        Возвращает кортеж (выходной_бит, цель) или (None, None), если цель не определена.
        """
        # Определяем входной бит: если inbox не пуст, берём его, иначе 0.
        x = self.inbox if self.inbox is not None else 0
        self.inbox = None  # сбрасываем после обработки

        # Сохраняем состояние до обновления, так как оно нужно для выбора цели
        s_old = self.state

        # Вычисляем выходной бит по правилу:
        # если состояние 0, выход = вход; если состояние 1, выход = инверсия входа.
        if s_old == 0:
            y = x
        else:
            y = 1 - x

        # Обновляем состояние: переключение (0 -> 1, 1 -> 0)
        self.state = 1 - s_old

        # Определяем цель (должно быть реализовано в подклассе)
        target = self._select_target(y, x, s_old)

        return y, target

    def _select_target(self, y, x, s_old):
        """
        Возвращает цель в формате (layer, local_id) или None.
        Должен быть переопределён в подклассах.
        y - выходной бит, x - входной бит, s_old - состояние до обновления.
        """
        raise NotImplementedError


class SimpleNeuron(Neuron):
    """
    Простой нейрон.
    Имеет два фиксированных адресата в следующем слое.
    Выходной бит y определяет, кому отправить сигнал:
        если y = 0 -> первому адресату (target0),
        если y = 1 -> второму адресату (target1).
    """
    def __init__(self, layer, target0, target1, initial_state=0):
        """
        target0, target1: кортежи (layer, local_id) адресатов (оба должны быть в слое layer+1).
        """
        super().__init__(layer, initial_state)
        self.targets = [target0, target1]

    def _select_target(self, y, x, s_old):
        # Простой нейрон не использует x и s_old для выбора цели, только y.
        return self.targets[y]


class AddressableNeuron(Neuron):
    """
    Адресный нейрон.
    Имеет:
      - два фиксированных адресата для обычного режима (fixed_targets),
      - список ровно из 4 адресов (address_list), каждый задаётся как (delta_layer, local_id),
      - таблицу действий (action_table) для каждой из четырёх комбинаций (s, x), где s - состояние до обновления.
    Действие может быть:
      - 'ordinary' – использовать обычный режим (выбрать цель из fixed_targets по выходному биту y),
      - ('address', i) – отправить по адресу address_list[i] (i = 0..3).
    """
    def __init__(self, layer, fixed_targets, address_list, action_table, initial_state=0):
        """
        fixed_targets: список из двух кортежей (layer, local_id) для обычного режима.
        address_list: список из четырёх кортежей (delta_layer, local_id). Смещение может быть любым целым,
                      целевой слой = текущий слой + delta_layer.
        action_table: словарь с ключами (s, x) -> действие. Пример: {(0,0): 'ordinary', (0,1): ('address',0), ...}
        """
        super().__init__(layer, initial_state)
        assert len(fixed_targets) == 2, "fixed_targets должен содержать ровно два адреса"
        assert len(address_list) == 4, "address_list должен содержать ровно четыре адреса"
        self.fixed_targets = fixed_targets
        self.address_list = address_list
        self.action_table = action_table

    def _select_target(self, y, x, s_old):
        # Извлекаем действие для данной комбинации (состояние до обновления, вход)
        action = self.action_table.get((s_old, x), 'ordinary')  # по умолчанию ordinary

        if action == 'ordinary':
            # Обычный режим: цель выбирается по выходному биту y из fixed_targets
            return self.fixed_targets[y]
        else:
            # Адресный режим: action = ('address', i)
            _, idx = action
            delta_layer, target_local = self.address_list[idx]
            target_layer = self.layer + delta_layer
            return (target_layer, target_local)


class Network:
    """
    Класс сети, содержащей нейроны.
    Обеспечивает:
      - добавление нейронов с автоматическим присвоением локальных ID в слоях,
      - преобразование локальных адресов (layer, local_id) в глобальные ID,
      - синхронный потактовый шаг с обработкой внешних входов и доставкой сигналов.
    """
    def __init__(self):
        self.neurons = []               # список всех нейронов (глобальный индекс = позиция в списке)
        self.layer_to_global = {}       # {layer: {local_id: global_id}}
        self.global_to_local = {}        # {global_id: (layer, local_id)}

    def add_neuron(self, neuron):
        """
        Добавляет нейрон в сеть и назначает ему локальный ID внутри его слоя.
        Возвращает глобальный ID нейрона.
        """
        global_id = len(self.neurons)
        self.neurons.append(neuron)

        layer = neuron.layer
        if layer not in self.layer_to_global:
            self.layer_to_global[layer] = {}
        local_id = len(self.layer_to_global[layer])   # следующий свободный локальный ID в этом слое
        self.layer_to_global[layer][local_id] = global_id
        self.global_to_local[global_id] = (layer, local_id)

        neuron.local_id = local_id   # сохраняем локальный ID в самом нейроне (для информации)
        return global_id

    def local_to_global(self, layer, local_id):
        """
        Преобразует пару (layer, local_id) в глобальный ID нейрона.
        Возвращает None, если такого нейрона нет.
        """
        return self.layer_to_global.get(layer, {}).get(local_id, None)

    def step(self, external_inputs=None):
        """
        Выполняет один такт работы сети.
        external_inputs: словарь {global_id: бит} для внешних входов (например, от датчиков).
        Возвращает список отправленных сигналов в формате (from_gid, to_gid, bit).
        """
        # 1. Применяем внешние входы: они добавляются к inbox соответствующих нейронов (OR)
        if external_inputs:
            for gid, bit in external_inputs.items():
                if 0 <= gid < len(self.neurons):
                    self.neurons[gid].receive(bit)

        # 2. Собираем исходящие сигналы от всех нейронов
        outgoing = []  # (from_gid, to_gid, bit)
        for gid, neuron in enumerate(self.neurons):
            y, target_local = neuron.step()   # target_local = (target_layer, target_local_id) или None
            if target_local is not None:
                target_layer, target_local_id = target_local
                target_gid = self.local_to_global(target_layer, target_local_id)
                if target_gid is not None:    # если адрес корректен
                    outgoing.append((gid, target_gid, y))

        # 3. Доставляем сигналы получателям (накопление в inbox на следующий такт)
        for from_gid, to_gid, bit in outgoing:
            self.neurons[to_gid].receive(bit)

        return outgoing