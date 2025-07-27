import os
import subprocess
import torch
from mealpy import FloatVar
from mealpy.bio_based import OriginalGWO
from mealpy.swarm_based import OriginalBBOA

# General settings
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 112 if device == "cuda" else 4
workers = 8 if device == "cuda" else 1
counter = 1  # Counter to distinguish run names

# Objective function that trains YOLOv10 and returns the final mAP@0.5:0.95 score
def objective_func(solution):
    global counter
    lr, lrf, wd, mo = solution
    print(f"🔧 Parameters: lr0={lr}, lrf={lrf}, wd={wd}, mo={mo}")

    # YOLOv10 training command
    train_command = [
        "yolo",
        "task=detect",
        "mode=train",
        "epochs=100",
        f"batch={batch_size}",
        f"workers={workers}",
        "imgsz=640",
        "plots=True",
        f"device={device}",
        f"model='/content/gdrive/MyDrive/YoLov10/yolov10/yolov10s.pt'",
        f"data='/content/gdrive/MyDrive/YoLov10/data.yaml'",
        f"name='GWO_BBOA_{counter}'",
        f"lr0={lr}",
        f"lrf={lrf}",
        f"weight_decay={wd}",
        f"momentum={mo}",
        "optimizer=AdamW"
    ]

    # Run the training process
    try:
        subprocess.run(train_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        return 0.0

    # Read the last value of mAP from results file
    try:
        with open('/content/gdrive/MyDrive/YoLov10/mAP50_95_results.txt', 'r') as f:
            lines = f.readlines()
            score = float(lines[-1].strip().split(':')[-1]) if lines else 0.0
    except Exception as e:
        print(f"Failed to read score: {e}")
        score = 0.0

    counter += 1
    print(f"mAP@0.5:0.95 = {score}")
    return score

# Define the problem search space
problem = {
    "obj_func": objective_func,
    "bounds": FloatVar(
        lb=[0.0001, 0.1, 0.0001, 0.85],   # lower bounds for [lr0, lrf, wd, momentum]
        ub=[0.01, 0.7, 0.001, 0.95]       # upper bounds
    ),
    "minmax": "max"
}

# === Phase 1: Run GWO to explore search space ===
print("Running GWO...")
gwo = OriginalGWO(epoch=5, pop_size=10)
gwo_result = gwo.solve(problem)
initial_population = [agent.solution for agent in gwo.pop]

# === Phase 2: Run BBOA using GWO population as initialization ===
print("\n Running BBOA using GWO output...")
bboa = OriginalBBOA(epoch=10, pop_size=10)
bboa.initial_pop = initial_population
bboa_result = bboa.solve(problem)

# === Final Result ===
print("\nFinal Result from Hybrid GWO-BBOA:")
print(f"Best solution: {bboa_result.solution}")
print(f"Best fitness (mAP@0.5:0.95): {bboa_result.target.fitness}")
