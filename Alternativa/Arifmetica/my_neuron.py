"""
my_neuron.py

Модифицированная версия с поддержкой двух режимов агрегации:
- mode=0 (OR): классическое поведение, inbox = OR всех пришедших сигналов.
- mode=1 (XOR): inbox = количество сигналов, входной бит = чётность количества.
"""

class Neuron:
    def __init__(self, layer, initial_state=0, mode=0):
        self.layer = layer
        self.local_id = None
        self.state = initial_state
        self.mode = mode          # 0=OR, 1=XOR
        if mode == 0:
            self.inbox = None     # для OR: None или бит
        else:
            self.inbox = 0        # для XOR: счётчик

    def receive(self, bit):
        if self.mode == 0:
            if self.inbox is None:
                self.inbox = bit
            else:
                self.inbox |= bit
        else:
            if bit == 1:
                self.inbox += 1

    def step(self):
        if self.mode == 0:
            x = self.inbox if self.inbox is not None else 0
            self.inbox = None
        else:
            x = self.inbox % 2
            self.inbox = 0

        s_old = self.state
        # классическое правило переключения
        if x == 0:
            y = s_old
            s_new = s_old
        else:
            s_new = 1 - s_old
            y = s_new

        self.state = s_new
        target_gid = self._select_target(y, x, s_old)
        return y, target_gid

    def _select_target(self, y, x, s_old):
        raise NotImplementedError


class SimpleNeuron(Neuron):
    def __init__(self, layer, initial_state=0, mode=0):
        super().__init__(layer, initial_state, mode)
        self.target_gids = [None, None]

    def _select_target(self, y, x, s_old):
        return self.target_gids[y]


class AddressableNeuron(Neuron):
    def __init__(self, layer, initial_state=0, mode=0):
        super().__init__(layer, initial_state, mode)
        self.fixed_gids = [None, None]
        self.address_gids = [None] * 4
        self.action_table = {}

    def _select_target(self, y, x, s_old):
        action = self.action_table.get((s_old, x), 'ordinary')
        if action == 'ordinary':
            return self.fixed_gids[y]
        else:
            _, idx = action
            if 0 <= idx < len(self.address_gids):
                return self.address_gids[idx]
            return None


class Network:
    def __init__(self):
        self.neurons = []
        self.layer_to_global = {}
        self.global_to_local = {}

    def add_neuron(self, neuron):
        global_id = len(self.neurons)
        self.neurons.append(neuron)
        layer = neuron.layer
        if layer not in self.layer_to_global:
            self.layer_to_global[layer] = {}
        local_id = len(self.layer_to_global[layer])
        self.layer_to_global[layer][local_id] = global_id
        self.global_to_local[global_id] = (layer, local_id)
        neuron.local_id = local_id
        return global_id

    def local_to_global(self, layer, local_id):
        return self.layer_to_global.get(layer, {}).get(local_id, None)

    def step(self, external_inputs=None):
        if external_inputs:
            for gid, bit in external_inputs.items():
                if 0 <= gid < len(self.neurons):
                    self.neurons[gid].receive(bit)
        outgoing = []
        for gid, neuron in enumerate(self.neurons):
            y, target_gid = neuron.step()
            if target_gid is not None and 0 <= target_gid < len(self.neurons):
                outgoing.append((gid, target_gid, y))
        for from_gid, to_gid, bit in outgoing:
            self.neurons[to_gid].receive(bit)
        return outgoing

    def reset_states(self, initial_state=0):
        for neuron in self.neurons:
            neuron.state = initial_state
            if neuron.mode == 0:
                neuron.inbox = None
            else:
                neuron.inbox = 0