
# YOLOv10 Hyperparameter Optimization for Arson Detection

This project implements hyperparameter tuning for the YOLOv10s model using four different metaheuristic algorithms to enhance arson detection performance:
- Particle Swarm Optimization (PSO)
- Grey Wolf Optimizer (GWO)
- Brown Bear Optimization Algorithm (BBOA)
- Hybrid GWO-BBOA (Proposed)

## Project Structure

Each algorithm is implemented in a separate script:
- `pso_yolov10.py` — PSO optimization
- `gwo_yolov10.py` — GWO optimization
- `bboa_yolov10.py` — BBOA optimization
- `hybrid_gwo_bboa.py` — Proposed hybrid optimization

## Requirements

- Python 3.8+
- YOLOv10 environment (`ultralytics` or custom YOLOv10 setup)
- [Mealpy](https://pypi.org/project/mealpy/)
- torch, wandb

# Environment Setup

We recommend using a virtual environment to keep dependencies isolated:

#Using `virtualenv`:

# Create and activate the environment
virtualenv yolov10-env
source yolov10-env/bin/activate  # On Windows: yolov10-env\Scripts\activate

# Install required libraries
pip install -r requirements.txt

# Dataset

The dataset used for training and evaluation is the Arson Detection Dataset, publicly available on Kaggle:
[Arson Dataset - Kaggle](https://www.kaggle.com/datasets/aliabbasabbod/arson-dataset)

# How to Run

Each script runs the corresponding optimization algorithm and sends parameters to YOLOv10 for training. Example:
```bash
python pso_yolov10.py
```

Make sure your paths in the script (to `model`, `data.yaml`, etc.) are correctly set for your environment.

## Output

Each run saves logs, best fitness, and optimized hyperparameters. Results are printed to the console.

## License

This project is for academic and research use only.
