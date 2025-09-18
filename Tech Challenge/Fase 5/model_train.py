from ultralytics import YOLO
import os
import numpy as np

def train(
    yolo_pt_path: str,
    data_yaml_path: str,
    train_dir_name: str,
    epochs: int = 100
) -> str:
    """
    Train a YOLO model and evaluate it on the test dataset.

    Args:
        yolo_weights_path (str): Path to the base YOLO .pt weights file (e.g., yolo11m.pt).
        data_yaml_path (str): Path to the dataset YAML config with 'train', 'val', and 'test' splits.
        train_experiment_name (str): Directory name under 'runs/train/' to save experiment outputs.
        epochs (int, optional): Number of training epochs. Defaults to 100.

    Returns:
        str: Path to the best checkpoint file (best.pt) after training.
    """
    # Launch training
    results = YOLO(yolo_pt_path).train(
        data            = data_yaml_path,
        name            = train_dir_name,
        epochs          = epochs,
        imgsz           = 640,            # Input image size for training
        multi_scale     = True,           # Random scale between 0.5x and 1.0x
        patience        = 10,             # Early stopping if no improvement for 40 epochs
        batch           = 8,
        workers         = 8,
        device          = 0,              # GPU device index
        augment         = True,           # Enable basic augmentations
        mosaic          = 1.0,            # Use mosaic augmentation (4 images)
        mixup           = 0.5,            # Use mixup augmentation
        translate       = 0.1,            # Random translation factor
        scale           = 0.5,            # Random scale factor
        perspective     = 0.0,            # Perspective augmentation
        optimizer       = 'AdamW',        # Optimizer choice
        lr0             = 0.0005,           # Initial learning rate
        lrf             = 0.05,           # Final learning rate factor (5% of lr0)
        warmup_epochs   = 3,              # Number of warmup epochs
        warmup_momentum = 0.8,            # Warmup momentum
        warmup_bias_lr  = 0.1,            # Warmup bias learning rate
    )
    print("🚀 Save dir:", results.save_dir)

    # Locate the best checkpoint
    trained_model_best_path = os.path.join(results.save_dir, "weights", "best.pt")
    if os.path.isfile(trained_model_best_path):
        print("✅ best.pt: ", trained_model_best_path)
    else:
        print("⚠️ best.pt not found: ", trained_model_best_path)

    #  Evaluate on the 'test' split and save JSON metrics
    val_results = YOLO(trained_model_best_path).val(
        data      = data_yaml_path,
        split     = "test",
        imgsz     = 640,
        batch     = 8,
        save_json = True
    )

    # Compute average metrics across all classes
    m = val_results.box
    p_mean = float(np.mean(m.p)) if hasattr(m.p, "mean") else float(m.p)
    r_mean = float(np.mean(m.r)) if hasattr(m.r, "mean") else float(m.r)
    map50_mean = float(np.mean(m.map50)) if hasattr(m.map50, "mean") else float(m.map50)
    map50_95_mean = float(m.map)
    # Display test metrics
    print("\n🎯 Test Metrics (mean per class):")
    print(f"  Precision:    {p_mean:.3f}")
    print(f"  Recall:       {r_mean:.3f}")
    print(f"  mAP@0.5:      {map50_mean:.3f}")
    print(f"  mAP@0.5:0.95: {map50_95_mean:.3f}")

    return trained_model_best_path
