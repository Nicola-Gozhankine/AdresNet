class Neuron:
    __slots__ = ('layer', 'local_id', 'state', 'mode', 'inbox')
    def __init__(self, layer, initial_state=0, mode=0):
        self.layer = layer
        self.local_id = None
        self.state = initial_state
        self.mode = mode          # 0 = OR, 1 = XOR
        self.inbox = 0            # для OR хранит бит, для XOR — счётчик

    def receive(self, bit):
        if self.mode == 0:
            self.inbox |= bit
        else:
            if bit:
                self.inbox += 1

    def step(self):
        if self.mode == 0:
            x = self.inbox
            self.inbox = 0
        else:
            x = self.inbox & 1    # чётность количества единиц
            self.inbox = 0

        s_old = self.state
        y = x ^ s_old
        self.state = 1 - s_old
        return y, self._select_target(y, x, s_old)

    def _select_target(self, y, x, s_old):
        raise NotImplementedError

class SimpleNeuron(Neuron):
    __slots__ = ('target_gids',)
    def __init__(self, layer, initial_state=0, mode=0):
        super().__init__(layer, initial_state, mode)
        self.target_gids = [None, None]

    def _select_target(self, y, x, s_old):
        return self.target_gids[y]

class AddressableNeuron(Neuron):
    __slots__ = ('fixed_gids', 'address_gids', 'action_list')
    def __init__(self, layer, initial_state=0, mode=0):
        super().__init__(layer, initial_state, mode)
        self.fixed_gids = [None, None]
        self.address_gids = [None] * 4
        self.action_list = ['ordinary'] * 4   # будет заполнено при построении

    def _select_target(self, y, x, s_old):
        idx = (s_old << 1) | x   # 0..3
        action = self.action_list[idx]
        if action == 'ordinary':
            return self.fixed_gids[y]
        else:  # action = 'addr0'..'addr3'
            return self.address_gids[int(action[4:])]