import numpy as np
import random
import json

NEURON_GENE_SIZE = 11

NUM_ACTIONS = 5
DELTA_RANGE = 2


# ------------------------------------------------
# CREATE RANDOM GENOTYPE
# ------------------------------------------------

def random_genotype(num_neurons):

    size = num_neurons * NEURON_GENE_SIZE

    return np.random.randint(
        0,
        2**31 - 1,
        size,
        dtype=np.int32
    )


# ------------------------------------------------
# CREATE FROM SEED
# ------------------------------------------------

def genotype_from_seed(seed, num_neurons):

    rng = np.random.default_rng(seed)

    size = num_neurons * NEURON_GENE_SIZE

    return rng.integers(
        0,
        2**31 - 1,
        size,
        dtype=np.int32
    )


# ------------------------------------------------
# MUTATION
# ------------------------------------------------

def mutate(genotype, rate=0.01, strength=10):

    mask = np.random.random(genotype.shape) < rate

    noise = np.random.randint(
        -strength,
        strength + 1,
        genotype.shape
    )

    genotype[mask] += noise[mask]

    return genotype


# ------------------------------------------------
# CROSSOVER
# ------------------------------------------------

def crossover(g1, g2):

    cut = random.randint(0, len(g1) - 1)

    child = np.empty_like(g1)

    child[:cut] = g1[:cut]
    child[cut:] = g2[cut:]

    return child


# ------------------------------------------------
# DECODE NEURON
# ------------------------------------------------

def decode_neuron(genotype, neuron_id, num_neurons):

    i = neuron_id * NEURON_GENE_SIZE

    g = genotype[i:i + NEURON_GENE_SIZE]

    t = g[0] % 3

    addr1 = g[1] % num_neurons
    addr2 = g[2] % num_neurons

    deltas = [
        (g[3] % 5) - 2,
        (g[4] % 5) - 2,
        (g[5] % 5) - 2,
        (g[6] % 5) - 2
    ]

    actions = [
        g[7] % NUM_ACTIONS,
        g[8] % NUM_ACTIONS,
        g[9] % NUM_ACTIONS,
        g[10] % NUM_ACTIONS
    ]

    return t, addr1, addr2, deltas, actions


# ------------------------------------------------
# EXPORT
# ------------------------------------------------

def save_genotype(genotype, path):

    with open(path, "w") as f:

        json.dump(genotype.tolist(), f)


# ------------------------------------------------
# LOAD
# ------------------------------------------------

def load_genotype(path):

    with open(path) as f:

        data = json.load(f)

    return np.array(data, dtype=np.int32)