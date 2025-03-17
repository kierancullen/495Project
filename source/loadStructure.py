from mpm import *
from LSystem import *
import jsonpickle
from pathlib import Path


file = "exampleStructure.json"
filePath = Path(file)

with open(file, "r") as f:
    json_str = f.read()
sys = jsonpickle.decode(json_str)

print("Loaded L-system:")
sys.print()
scoreStructure(sys.getIteration(), "structures", filePath.stem, vis=True, render=True, renderFrameLimit = 5000, crop=False)