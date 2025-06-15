from ultralytics import YOLO
import torch
import os
from pathlib import Path

def train_model(
    data_yaml_path="dataset/dataset.yaml",
    epochs=50,
    batch_size=16,
    img_size=640,
    device="cuda"
):
    """
    Train a YOLOv8 model on the Star Wars character dataset.
    
    Args:
        data_yaml_path (str): Path to the dataset configuration file
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Input image size
        device (str): Device to use for training ('cuda' for GPU)
    """
    print("Starting Star Wars character detection model training...")
    
    # Verificar si CUDA está disponible
    if not torch.cuda.is_available():
        print("WARNING: CUDA no está disponible. El entrenamiento será más lento en CPU.")
        device = "cpu"
    else:
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    
    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        patience=10,  # Early stopping patience
        save=True,    # Save best model
        project="runs/detect",
        name="star_wars_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=True
    )
    
    # Print training results
    print("\nTraining completed!")
    print(f"Best model saved at: {results.save_dir}")
    
    # Validate the model
    print("\nValidating model...")
    metrics = model.val()
    print(f"Validation mAP50: {metrics.box.map50:.3f}")
    print(f"Validation mAP50-95: {metrics.box.map:.3f}")
    
    return model

if __name__ == "__main__":
    # Ensure the dataset exists
    data_yaml = Path("dataset/dataset.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(
            "Dataset not found. Please run the data preparation script first: "
            "python data_preparation/3_prepare_dataset.py"
        )
    
    # Train the model
    model = train_model()
    
    # Save the model in ONNX format for deployment
    model.export(format="onnx")
    print("\nModel exported to ONNX format for deployment") 