from mpm import *
from LSystem import *
import jsonpickle
import math
import os

def exportJson(lsystem, filename):
    json_str = jsonpickle.encode(lsystem)
    with open(filename, "w") as f:
        f.write(json_str)

generationSize = 10
split = 0.1
generations = 2

currentScores = []
currentStructures = []
genTopDeciles = []
genMaxes = []

os.makedirs("structures", exist_ok=True)

for i in range(generationSize):
    newSystem = LSystem([],[])
    newSystem.seed()
    currentStructures.append(newSystem)

for i in range(generations):
    currentGenerationSize = len(currentStructures)
    #Score the structures
    for j in range(currentGenerationSize):
        print(f"generation {i}, structure {j}")
        exportJson(currentStructures[j], f"structures/gen{i}_{j}.json")
        score = scoreStructure(currentStructures[j].getIteration(), "structures", f"gen{i}_{j}")
        currentScores.append(score) if not math.isnan(score) else currentScores.append(-10000)
        exportJson(currentStructures[j], f"structures/gen{i}_{j}_{currentScores[-1]:.3f}.json")

    #Determine the cutoff
    cutoffIndex = int(currentGenerationSize*(1-split))
    cutoffScore = sorted(currentScores)[cutoffIndex]
    print(f"Cutoff score for generation {i}: {cutoffScore:.3f}")
    print(f"Top score for generation {i}: {max(currentScores):.3f}")
    genTopDeciles.append(cutoffScore)
    genMaxes.append(max(currentScores))

    #Create the new generation
    newStructures = []
    for j in range(currentGenerationSize):
        if currentScores[j]>=cutoffScore:
            for _ in range(int(1.0/split)):
                newStructures.append(currentStructures[j].clone())
                newStructures[-1].mutate()
                newStructures[-1].ancestor = f"gen{i}_{j}"
    
    currentStructures = newStructures
    currentScores = []

gens = list(range(0, generations))

fig, ax = plt.subplots()

ax.plot(gens, genMaxes, label="top score", color="orange", linestyle="-")

ax.plot(gens, genTopDeciles, label="top decile", color="gray", linestyle="-")

ax.set_xlabel("generation")
ax.set_ylabel("score")
ax.set_xticks(range(min(gens), max(gens) + 1))

ax.legend()

plt.savefig("structures/plot.svg", format="svg")
