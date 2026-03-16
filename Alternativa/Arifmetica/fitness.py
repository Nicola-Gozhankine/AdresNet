import network_core
from network import *

MAX_STEPS = 200
MAX_QUEUE = 200


# ------------------------------------------------
# RUN NETWORK WITH METRICS
# ------------------------------------------------

def run_network(net, inputs):

    result = network_core.run_network_fast(
        net.ntype,
        net.state,
        net.param,
        net.out_index,
        net.conn_to,
        inputs
    )

    if result is None:
        return None, 0, 0

    return result


# ------------------------------------------------
# FITNESS
# ------------------------------------------------

def evaluate(genotype, task):

    net = Network()
    net.build_from_genotype(genotype)

    correct = 0
    total_steps = 0
    total_queue = 0

    for inputs, expected in task:

        net.reset_state()

        out, steps, queue_max = run_network(net, inputs)

        if out is None:
            return -1.0

        if out == expected:
            correct += 1

        total_steps += steps
        total_queue += queue_max

    accuracy = correct / len(task)

    # сложность сети
    complexity = net.n + len(net.conn_from)

    cost = (
        total_steps * 2
        + total_queue * 4
        + complexity * 3
    )

    fitness = accuracy - 0.001 * cost

    return fitness