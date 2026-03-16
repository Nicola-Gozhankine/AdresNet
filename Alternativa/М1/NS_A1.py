import itertools

class Neuron:
    """Базовый класс нейрона."""
    def __init__(self, layer, initial_state=0):
        self.layer = layer
        self.local_id = None
        self.state = initial_state
        self.inbox = None

    def receive(self, bit):
        if self.inbox is None:
            self.inbox = bit
        else:
            self.inbox |= bit

    def step(self):
        x = self.inbox if self.inbox is not None else 0
        self.inbox = None
        s_old = self.state
        # выходной бит
        y = x if s_old == 0 else 1 - x
        # обновление состояния
        self.state = 1 - s_old
        target = self._select_target(y, x, s_old)
        return y, target

    def _select_target(self, y, x, s_old):
        raise NotImplementedError

    def __repr__(self):
        return f"N(layer={self.layer}, local={self.local_id}, state={self.state})"


class SimpleNeuron(Neuron):
    def __init__(self, layer, target0, target1, initial_state=0):
        super().__init__(layer, initial_state)
        self.targets = [target0, target1]

    def _select_target(self, y, x, s_old):
        return self.targets[y]


class AddressableNeuron(Neuron):
    def __init__(self, layer, fixed_targets, address_list, action_table, initial_state=0):
        super().__init__(layer, initial_state)
        assert len(fixed_targets) == 2
        assert len(address_list) == 4
        self.fixed_targets = fixed_targets
        self.address_list = address_list
        self.action_table = action_table

    def _select_target(self, y, x, s_old):
        action = self.action_table.get((s_old, x), 'ordinary')
        if action == 'ordinary':
            return self.fixed_targets[y]
        else:  # ('address', i)
            _, idx = action
            delta, target_local = self.address_list[idx]
            target_layer = self.layer + delta
            return (target_layer, target_local)


class Network:
    def __init__(self):
        self.neurons = []
        self.layer_to_global = {}
        self.global_to_local = {}

    def add_neuron(self, neuron):
        gid = len(self.neurons)
        self.neurons.append(neuron)
        layer = neuron.layer
        if layer not in self.layer_to_global:
            self.layer_to_global[layer] = {}
        local_id = len(self.layer_to_global[layer])
        self.layer_to_global[layer][local_id] = gid
        self.global_to_local[gid] = (layer, local_id)
        neuron.local_id = local_id
        return gid

    def local_to_global(self, layer, local_id):
        return self.layer_to_global.get(layer, {}).get(local_id, None)

    def step(self, external_inputs=None):
        # внешние входы
        if external_inputs:
            for gid, bit in external_inputs.items():
                if 0 <= gid < len(self.neurons):
                    self.neurons[gid].receive(bit)

        # сбор исходящих сигналов
        outgoing = []  # (from_gid, to_gid, bit)
        for gid, neuron in enumerate(self.neurons):
            y, target_local = neuron.step()
            if target_local is not None:
                t_layer, t_local = target_local
                t_gid = self.local_to_global(t_layer, t_local)
                if t_gid is not None:
                    outgoing.append((gid, t_gid, y))

        # доставка
        for from_gid, to_gid, bit in outgoing:
            self.neurons[to_gid].receive(bit)

        return outgoing

    def print_state(self):
        """Вывести текущее состояние всех нейронов."""
        for gid, n in enumerate(self.neurons):
            print(f"Нейрон {gid} (слой {n.layer}.{n.local_id}): состояние={n.state}, inbox={n.inbox}")


# ========== Примеры для экспериментов ==========

def example_1():
    """Простейшая сеть: входной нейрон -> один скрытый -> выходной адресный."""
    net = Network()

    # слой 0: входной простой нейрон
    in0 = SimpleNeuron(layer=0, target0=(1,0), target1=(1,0))  # оба выхода в один и тот же нейрон слоя 1
    in_id = net.add_neuron(in0)

    # слой 1: простой нейрон
    hidden = SimpleNeuron(layer=1, target0=(2,0), target1=(2,0))
    h_id = net.add_neuron(hidden)

    # слой 2: адресный нейрон
    fixed = [(1,0), (1,0)]  # обычный режим ведёт обратно в hidden
    addr = [(0,0), (0,0), (0,0), (0,0)]  # все адреса ведут на самого себя (для демонстрации)
    # action: при (0,0) и (0,1) обычный, при (1,0) и (1,1) адресный (но все адреса одинаковы)
    action = {(0,0): 'ordinary', (0,1): 'ordinary', (1,0): ('address',0), (1,1): ('address',1)}
    out = AddressableNeuron(layer=2, fixed_targets=fixed, address_list=addr, action_table=action)
    out_id = net.add_neuron(out)

    print("Состояние до тактов:")
    net.print_state()

    for t in range(5):
        print(f"\n--- Такт {t} ---")
        # подаём вход: чередуем 1 и 0
        ext = {in_id: (t % 2)}
        outgoing = net.step(ext)
        print("Отправленные сигналы:", [(f"н{from_}", f"н{to}", bit) for from_, to, bit in outgoing])
        net.print_state()


def example_2():
    """Более сложная сеть с двумя скрытыми нейронами и обратными связями."""
    net = Network()

    # слой 0: входной простой нейрон (ведёт на оба нейрона слоя 1)
    in0 = SimpleNeuron(layer=0, target0=(1,0), target1=(1,1))
    in_id = net.add_neuron(in0)

    # слой 1: два простых нейрона (ведут на выходной слой 2)
    h0 = SimpleNeuron(layer=1, target0=(2,0), target1=(2,0))
    h1 = SimpleNeuron(layer=1, target0=(2,0), target1=(2,0))
    net.add_neuron(h0)
    net.add_neuron(h1)

    # слой 2: адресный нейрон
    fixed = [(1,0), (1,1)]  # обычный режим ведёт на h0 и h1
    # адреса: 
    # 0: на h0 (смещение -1)
    # 1: на h1 (смещение -1)
    # 2: на входной нейрон (смещение -2)
    # 3: на самого себя (смещение 0)
    addr = [(-1,0), (-1,1), (-2,0), (0,0)]
    # action_table: 
    # (0,0): ordinary
    # (0,1): ordinary
    # (1,0): address 0 (послать на h0)
    # (1,1): address 1 (послать на h1)
    action = {(0,0): 'ordinary', (0,1): 'ordinary', (1,0): ('address',0), (1,1): ('address',1)}
    out = AddressableNeuron(layer=2, fixed_targets=fixed, address_list=addr, action_table=action)
    out_id = net.add_neuron(out)

    print("Состояние до тактов:")
    net.print_state()

    # подаём последовательность входов: 1,0,1,0,1
    inputs = [1,0,1,0,1]
    for t, bit in enumerate(inputs):
        print(f"\n--- Такт {t} (вход={bit}) ---")
        outgoing = net.step({in_id: bit})
        print("Отправленные сигналы:", [(f"н{from_}", f"н{to}", bit) for from_, to, bit in outgoing])
        net.print_state()


if __name__ == "__main__":
    print("=== Пример 1 ===")
    example_1()
    print("\n=== Пример 2 ===")
    example_2() 