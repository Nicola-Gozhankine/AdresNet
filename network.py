from collections import deque

class Network:
    __slots__ = ('neurons', 'layer_to_global', '_queue', '_in_queue')

    def __init__(self):
        self.neurons = []
        self.layer_to_global = {}
        self._queue = deque()
        self._in_queue = set()

    def add_neuron(self, neuron):
        gid = len(self.neurons)
        self.neurons.append(neuron)
        layer = neuron.layer
        if layer not in self.layer_to_global:
            self.layer_to_global[layer] = {}
        local_id = len(self.layer_to_global[layer])
        self.layer_to_global[layer][local_id] = gid
        neuron.local_id = local_id
        return gid

    def local_to_global(self, layer, local_id):
        return self.layer_to_global.get(layer, {}).get(local_id)

    def _enqueue(self, gid):
        if gid not in self._in_queue:
            self._queue.append(gid)
            self._in_queue.add(gid)

    def external_input(self, gid, bit):
        if 0 <= gid < len(self.neurons):
            self.neurons[gid].receive(bit)
            self._enqueue(gid)

    def step(self):
        """Обрабатывает ровно один нейрон из очереди."""
        if not self._queue:
            return False
        gid = self._queue.popleft()
        self._in_queue.discard(gid)
        neuron = self.neurons[gid]
        y, target = neuron.step()
        if target is not None and 0 <= target < len(self.neurons):
            self.neurons[target].receive(y)
            self._enqueue(target)
        return True

    def reset(self):
        for n in self.neurons:
            n.state = 0
            n.inbox = 0
        self._queue.clear()
        self._in_queue.clear()

    def is_quiet(self):
        return not self._queue