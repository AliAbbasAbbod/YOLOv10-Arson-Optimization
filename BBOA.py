import os
import subprocess
import pandas as pd
from mealpy import FloatVar, BBOA
import torch
import json
# عداد لتغيير اسم المشروع
counter = 1


# إعداد الجهاز بناءً على توفر CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
batch_size = 112 if device == "cuda" else 4
workers = 8 if device == "cuda" else 1

import wandb
wandb.require("legacy-service")



# دالة الهدف
def objective_func(solution):
    global counter
    lr, lrf, wd, mo = solution
    print(f"Solution passed to YOLO: lr0={lr}, lrf={lrf}, weight_decay={wd}, momentum={mo}")

    # أمر التدريب باستخدام YOLO
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
        f"name='BBOA_{counter}'",
        f"lr0={lr}",
        f"lrf={lrf}",
        f"weight_decay={wd}",
        f"momentum={mo}",
        "optimizer=AdamW"
    ]
    
    subprocess.run(train_command, check=True)

    # قراءة النتائج من ملف النصوص

    try:
        with open('/content/gdrive/MyDrive/YoLov10/mAP50_95_results.txt', 'r') as f:
            lines = f.readlines()
            last_value = float(lines[-1].strip().split(':')[-1]) if lines else 0.0
    except Exception as e:
        print(f"Error reading results: {e}")
        last_value = 0.0

    counter += 1
    return last_value



problem_dict = {
    "obj_func": objective_func,  # دالة الهدف
    "bounds": FloatVar(lb=[0.0001, 0.1, 0.0001, 0.85], ub=[0.01, 0.7, 0.001, 0.95]),
    "minmax": "max",  # نوع التحسين (تعظيم أو تصغير)
}

# الدالة الرئيسية
def main():
    # تشغيل BBOA للبحث عن الحل الأمثل
    print("\n=== Running BBOA for Optimization ===")
    model = BBOA.OriginalBBOA(epoch=10, pop_size=50)
    g_best = model.solve(problem_dict)
    print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

if __name__ == "__main__":
    main()