import os
import sys
import subprocess
from mealpy import FloatVar, PSO
# عدد العصور التي يمكن أن تمر بدون أي تحسين قبل التوقف
early_stopping_patience = 5

# عداد لتغيير اسم المشروع
counter = 1
os.environ["WANDB_MODE"] = "disabled"

def objective_func(solution):
    global counter  # استخدام المتغير العام counter

    lr, lrf, wd, mo = solution
    
    # إعداد الأمر لتدريب النموذج مع البراميترات التي تم تحسينها
    train_command = [
        "yolo", 
        "task=detect", 
        "mode=train", 
        "epochs=150", 
        "batch=112",
        "workers=12",  # هنا تم إضافة عدد العمال
        "imgsz=640", 
        "plots=True", 
        "device=0",  # استخدام أكثر من GPU
        f"model='/content/gdrive/MyDrive/YoLov10/yolov10/yolov10s.pt'",
        f"data='/content/gdrive/MyDrive/yolov9/data/data.yaml'",
        f"project='GWOSARO'",  # اسم مشروع بسيط
        f"name='GWOSARO_{counter}'",
        "exist_ok=True",
        f"lr0={lr}", 
        f"lrf={lrf}", 
        f"weight_decay={wd}", 
        f"momentum={mo}",
        "optimizer=AdamW"
    ]
    # تنفيذ الأمر باستخدام subprocess
    subprocess.run(train_command, check=True)


      # فتح الملف في وضع القراءة
    with open('/content/gdrive/MyDrive/YoLov10/mAP50_95_results.txt', 'r') as f:
        # قراءة جميع الأسطر من الملف
        lines = f.readlines()

        # التحقق من أن الملف ليس فارغًا
        if lines:
        # الحصول على آخر سطر
            last_line = lines[-1]
        else:
            print("الملف فارغ.")
    # طباعة آخر قيمة مع التأكد من أنها ليست فارغة
    if last_line:
        try:
        # تحويل السطر الأخير إلى قيمة عشرية
          last_value = float(last_line.strip().split(':')[-1])  # الحصول على الجزء العددي فقط
        except ValueError as e:
          print(f"خطأ: تعذر تحويل السطر الأخير إلى قيمة عشرية. السطر: {last_line}")
    else:
      last_value = None
    
    print(last_value)
    counter += 1            # زيادة العداد بعد كل تشغيل

    return  last_value

# متغيرات التوقف المبكر
best_fitness = -float('inf')  # نبدأ بأقل قيمة ممكنة
no_improvement_epochs = 0  # عداد لعدد العصور بدون تحسين

problem_dict = {
    "obj_func": objective_func,
    "bounds": FloatVar(lb=[0.0001, 0.1, 0.0001, 0.85], ub=[0.01, 0.7, 0.001, 0.95]),
    "minmax": "max",
}

optimizer = PSO.OriginalPSO(epoch=20, pop_size=15)

# حل المشكلة مع مراقبة شرط التوقف المبكر
for epoch in range(optimizer.epoch):
    print(f"\n==== Epoch {epoch + 1}/{optimizer.epoch} ====")

    optimizer.solve(problem_dict)
    for idx, individual in enumerate(optimizer.pop):
        print(f"Individual {idx + 1}: Solution = {individual.solution}, Fitness = {individual.fitness}")

    # إذا كانت أفضل دقة تحسنت
    if optimizer.g_best.target.fitness > best_fitness:
        print(f"✅ Performance improved: {best_fitness} -> {optimizer.g_best.target.fitness}")
        best_fitness = optimizer.g_best.target.fitness
        no_improvement_epochs = 0  # إعادة ضبط العداد لأن هناك تحسين
    else:
        no_improvement_epochs += 1  # لم يكن هناك تحسين
        print(f"⚠️ No improvement in this epoch. Consecutive no improvement epochs: {no_improvement_epochs}")

    
    print(f"Best Fitness so far: {best_fitness}")
    print(f"Fitness this epoch: {optimizer.g_best.target.fitness}")
    
    # التحقق من التوقف المبكر
    if no_improvement_epochs >= early_stopping_patience:
        print(f"⛔ Early stopping: No improvement after {no_improvement_epochs} consecutive epochs.")
        break  # الخروج من الحلقة لأن التوقف المبكر تحقق
    
print("Best Solution: ", optimizer.g_best.solution)
print("Best Accuracy (Fitness): ", optimizer.g_best.target.fitness)
