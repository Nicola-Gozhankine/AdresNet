import time
import random
import numpy as np


# ==== ИМПОРТ ТВОЕЙ МОДЕЛИ ====
from genotype import *
from network import *
from fitness import *

# ==== ЛУЧШИЙ ГЕНОТИП ====
best_genotype = np.array(  [1379269001,  632432780,  566141958, 1870434849, 1967792035,
        163591707,  336974528,  760315553, 1514196811, 1178471954,
        766839712, 1646177587, 1268027407, 1272148667, 1705466018,
        400033858,  608950464,   23599074, 1144377148,  814624999,
       1959882244, 1348612881, 1186376496,  496270170, 1945768815,
       1226052778, 2108221078, 1168602398, 1495677401, 2103704167,
       1572327949,  495220276, 1378512602,   18414736, 1434158539,
       1732525248, 1182315774, 1037993099,  713667236,  585142849,
       1197634843, 1523950495, 1574505052, 1687865580, 1776594941,
        531364307, 1166730188,   76027255,  524917582, 1186647667,
         89974393, 1465402765,  439564135,  418008779, 2114744426,
       1061930539,  958513096,  832689731,  958886714, 1002511604,
       2123539985,  781439382, 1022022019, 1915009348,  196362733,
       1400454018], dtype=np.int32)

task = "1101"

# ==== СТРОИМ ТВОЮ СЕТЬ ====
net = Network()
net.build_from_genotype({"genotype": best_genotype})

# -----------------------------
# тестовые входы
# -----------------------------

inputs = [random.randint(0,3) for _ in range(100000)]

# -----------------------------
# BitNet speed test
# -----------------------------

start = time.time()

correct = 0

for x in inputs:
    net.reset_state()
    b0 = x & 1
    b1 = (x >> 1) & 1

    inp = [(0,b0),(1,b1)]

    out,_,_ = run_network(net,inp)

    if out is not None:
        out = out & 1

    target = [1,1,0,1][x]

    if out == target:
        correct += 1

end = time.time()

bitnet_time = (end-start)/len(inputs)*1e6
bitnet_acc = correct/len(inputs)

# -----------------------------
# RANN
# -----------------------------

print("Training RANN for 60 seconds...")

W1 = np.random.randn(2,8)
W2 = np.random.randn(8,1)

lr = 0.1
start = time.time()

while time.time()-start < 60:

    x = random.randint(0,3)

    b0 = x & 1
    b1 = (x>>1)&1

    inp = np.array([b0,b1])

    target = [1,1,0,1][x]

    h = np.maximum(0, inp@W1)
    out = h@W2

    pred = 1 if out>0 else 0

    err = target - pred

    W2 += lr * err * h.reshape(-1,1)
    W1 += lr * err * inp.reshape(-1,1)

print("Training finished")

# -----------------------------
# RANN benchmark
# -----------------------------

start = time.time()

correct = 0

for x in inputs:

    b0 = x & 1
    b1 = (x>>1)&1

    inp = np.array([b0,b1])

    h = np.maximum(0, inp@W1)
    out = h@W2

    pred = 1 if out>0 else 0

    target = [1,1,0,1][x]

    if pred == target:
        correct += 1

end = time.time()

rann_time = (end-start)/len(inputs)*1e6
rann_acc = correct/len(inputs)

# -----------------------------
# RESULTS
# -----------------------------

print()
print("RESULTS")
print("=================================")
print(f"AdresNet  accuracy: {bitnet_acc:.3f}   speed: {bitnet_time:.2f} us")
print(f"RANN    accuracy: {rann_acc:.3f}   speed: {rann_time:.2f} us")