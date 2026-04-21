from ultralytics import YOLO
from dotenv import load_dotenv
import os
import wandb

# Load environment variables from .env file
load_dotenv()

# Verify critical environment variables exist
project_id = os.getenv("WANDB_PROJECT")
if not project_id:
    raise ValueError("WANDB_PROJECT not found in .env file")

print("Environment loaded successfully")
print(f"Logging to W&B project: {project_id}")

# Initialize wandb experiment tracking
wandb.init(
    project=os.getenv("WANDB_PROJECT"),
    name="yolov8n-baseline-run1",
    config={
        "model": "yolov8n",
        "epochs": 50,
        "batch_size": 16,
        "image_size": 640,
        "dataset": "UA-DETRAC-140k"
    }
)

print("Weights & Biases initialized")
print("Training run: yolov8n-baseline-run1")

# Load YOLOv8 nano — smallest model, use this for baseline first
# nano trains fast and confirms everything works before running medium
model = YOLO('yolov8n.pt')

print("Model loaded: YOLOv8 nano")
print("Starting training...")
print("This will take several hours on CPU — let it run overnight")

# Train the model
results = model.train(
    data='/Users/newowner/Projects/highway-safety-ai/data/raw/ua-detrac/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='highway_safety_v1',
    project='runs/train',
    patience=10,
    save=True,
    plots=True
)

# Print results
print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)

map50 = results.results_dict.get('metrics/mAP50(B)', 0)
map95 = results.results_dict.get('metrics/mAP50-95(B)', 0)

print(f"mAP@0.5:     {map50:.4f}")
print(f"mAP@0.5-0.95: {map95:.4f}")
print(f"Model saved to: runs/train/highway_safety_v1/weights/best.pt")

wandb.log({
    "final_mAP50": map50,
    "final_mAP50-95": map95
})

wandb.finish()
print("Results logged to Weights & Biases")
