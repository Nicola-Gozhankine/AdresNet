# network.py
from collections import deque
import network_core

class Network:
    """
    Быстрая event-driven сеть.
    Генотип -> сеть -> симуляция.
    """

    def __init__(self):

        self.n = 0

        self.state = []
        self.ntype = []
        self.param = []

        self.conn_from = []
        self.conn_to = []

        self.out_index = []

    # -------------------------------------------------------
    # BUILD NETWORK
    # -------------------------------------------------------

    def build_from_genotype(self, genotype):

        g = genotype["genotype"]

        p = 0

        # --- защита от выхода за границы генома ---
        max_neurons = max(4, (len(g) - 1) // 2)

        self.n = max(4, abs(g[p]) % max_neurons)
        p += 1

        # состояния
        self.state = [0] * self.n

        # параметры
        self.ntype = []
        self.param = []

        for _ in range(self.n):

            if p >= len(g):
                break

            t = abs(g[p]) % 3
            p += 1

            if p >= len(g):
                break

            param = g[p]
            p += 1

            self.ntype.append(t)
            self.param.append(param)

        # если генотип кончился раньше
        while len(self.ntype) < self.n:
            self.ntype.append(0)
            self.param.append(0)

        # связи
        self.conn_from = []
        self.conn_to = []

        while p + 1 < len(g):

            src = abs(g[p]) % self.n
            dst = abs(g[p + 1]) % self.n

            p += 2

            if src != dst:
                self.conn_from.append(src)
                self.conn_to.append(dst)

        # индекс исходящих
        self.out_index = [[] for _ in range(self.n)]

        for i in range(len(self.conn_from)):

            src = self.conn_from[i]

            self.out_index[src].append(i)

    # -------------------------------------------------------
    # RESET
    # -------------------------------------------------------

    def reset(self):

        for i in range(self.n):
            self.state[i] = 0

    # -------------------------------------------------------
    # SIMULATION
    # -------------------------------------------------------

    def run(self, inputs, max_steps=64):

        queue = deque()

        for idx, val in inputs:

            if idx < self.n:
                self.state[idx] = val
                queue.append(idx)

        steps = 0

        while queue and steps < max_steps:

            neuron = queue.pop()
            

            self._process(neuron, queue)

            steps += 1

    # -------------------------------------------------------
    # NEURON PROCESS
    # -------------------------------------------------------

    def _process(self, i, queue):

        network_core.process(
            i,
            self.ntype,
            self.state,
            self.param,
            self.out_index,
            self.conn_to,
            queue
        )
    # -------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------

    def read(self, neurons):

        result = []

        for n in neurons:

            if n < self.n:
                result.append(self.state[n] & 1)
            else:
                result.append(0)

        return result
    


    def run_trace(self, inputs, max_steps=64):

        print("\n--- TRACE START ---")

        queue = []

        # подаем вход
        for idx, val in inputs:

            if idx >= self.n:
                continue

            self.state[idx] = val
            queue.append(idx)

            print(f"INPUT neuron {idx} = {val}")

        step = 0

        while queue and step < max_steps:

            neuron = queue.pop()

            print("\nSTEP", step)
            print("process neuron:", neuron)
            print("state before:", self.state)

            # ---- обычная обработка ----
            t = self.ntype[neuron]
            s = self.state[neuron]
            p = self.param[neuron]

            new_state = s

            if t == 0:
                new_state = s ^ (p & 3)

            elif t == 1:
                new_state = (s + p) & 3

            elif t == 2:
                new_state = (s * (p | 1)) & 3

            print("type:", t, "param:", p)
            print("new_state:", new_state)

            if new_state != s:

                self.state[neuron] = new_state

                for edge in self.out_index[neuron]:

                    dst = self.conn_to[edge]

                    print("send ->", dst)

                    self.state[dst] ^= new_state
                    queue.append(dst)

            print("state before:", self.state)
            print("queue:", queue)

            step += 1

        print("\n--- TRACE END ---")


    def _process_trace(self, i, queue):

        t = self.ntype[i]
        s = self.state[i]
        p = self.param[i]

        print(f"type={t} state={s} param={p}")

        new_state = s

        if t == 0:
            new_state = s ^ (p & 3)

        elif t == 1:
            new_state = (s + p) & 3

        elif t == 2:
            new_state = (s * (p | 1)) & 3

        print(f"new_state={new_state}")

        if new_state == s:
            print("no change")
            return

        self.state[i] = new_state

        for edge in self.out_index[i]:

            dst = self.conn_to[edge]

            print(f"send → {dst}")

            self.state[dst] ^= new_state

            queue.append(dst)
 
    
    
    
    
    def reset_state(self):
        self.state = [0] * self.n
    






 # 
 # 
 #  
    



#_______________________________________________________________
