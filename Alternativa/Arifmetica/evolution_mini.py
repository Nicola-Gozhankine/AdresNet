import random
import time
from copy import deepcopy
import network_core
print(dir(network_core))
import network_core

# core
import numpy as np
import random
import time

# project
from genotype import *
from network import *
from fitness import *

# -------------------------
# параметры
# -------------------------

POP_SIZE = 300
GENERATIONS = 1000

ELITE = 10
TOURNAMENT = 5

MUTATION_RATE = 0.2

NUM_NEURONS = 6


# -------------------------
# турнирный отбор
# -------------------------

def tournament(pop):

    best = None

    for _ in range(TOURNAMENT):

        g = random.choice(pop)

        if best is None or g["fitness"] > best["fitness"]:
            best = g

    return best


# -------------------------
# основная эволюция
# -------------------------

def evolve(task):

    start = time.time()

    population = []

    # -------------------------
    # начальная популяция
    # -------------------------

    for _ in range(POP_SIZE):

        g = random_genotype(NUM_NEURONS)

        population.append({
            "genotype": g,
            "fitness": evaluate({"genotype": g}, task)
        })

    best = None


    # -------------------------
    # поколения
    # -------------------------

    for gen in range(GENERATIONS):

        population.sort(key=lambda x: x["fitness"], reverse=True)

        if best is None or population[0]["fitness"] > best["fitness"]:
            best = deepcopy(population[0])

        avg = sum(p["fitness"] for p in population) / POP_SIZE

        print(
            f"gen {gen:4d} "
            f"best={population[0]['fitness']:.4f} "
            f"avg={avg:.4f}"
        )

        new_pop = []

        # -------------------------
        # элита
        # -------------------------

        for i in range(ELITE):

            new_pop.append({
                "genotype": population[i]["genotype"].copy(),
                "fitness": population[i]["fitness"]
            })

        # -------------------------
        # размножение
        # -------------------------

        while len(new_pop) < POP_SIZE:

            p1 = tournament(population)
            p2 = tournament(population)

            child = crossover(
                p1["genotype"],
                p2["genotype"]
            )

            if random.random() < MUTATION_RATE:

                child = mutate(child.copy())

            fitness = evaluate({"genotype": child}, task)

            new_pop.append({
                "genotype": child,
                "fitness": fitness
            })

        population = new_pop


    end = time.time()

    best["task"] = task
    best["generations"] = GENERATIONS
    best["pop_size"] = POP_SIZE
    best["time_sec"] = end - start

    return best




# -------------------------
# запуск
# -------------------------

if __name__ == "__main__":

    task = [

    ([(0,0),(1,0)], 1),
    ([(0,1),(1,0)], 1),
    ([(0,0),(1,1)], 0),
    ([(0,1),(1,1)], 1)

    ]

    result = evolve(task)
    best = result["genotype"]

    net = Network()
    net.build_from_genotype({"genotype": best})

    print("RESULT TABLE")

    for i in range(4):

        b0 = i & 1
        b1 = (i >> 1) & 1

        inp = [(0,b0),(1,b1)]

        net.reset()

        out,_,_ = run_network(net, inp)

        print(i, "->", out, "->", out & 1)

    print("\nRESULT\n")
    print(result)

    net.reset()

    net.run_trace([(0,1)])

    print("neurons:", net.n)
    print("connections:", len(net.conn_to))