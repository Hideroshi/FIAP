import json
import os

from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List, Union, Any
from PIL import Image


def predict(
    trained_dir_name: str,
    trained_model_file_path: str,
    source_file_path: Union[str, Image.Image] = None,
    conf: float = 0.5
) -> tuple[List[Results], List[Any]]:
    """
    Run inference using a trained YOLO model on a single image,
    save bounding-box images and text output, and generate JSON result files.

    Args:
        trained_dir_name (str): Name of the directory under `data/reports/` to save JSON outputs.
        trained_model_file_path (str): Path to the trained YOLO `best.pt` weights file.
        source_file_path (str): Path to the input image for inference.
        conf (float, optional): Minimum confidence threshold for detections. Defaults to 0.5.

    Returns:
        List[Results]: List of Ultralytics `Results` objects containing detection outputs.
        List[Any]: Report JSON Metadata.
    """
    # Perform prediction (inference) on the image
    results: List[Results] = YOLO(trained_model_file_path).predict(
        source     = source_file_path, # Input image file
        conf       = conf,             # Confidence threshold filter
        iou        = 0.45,             # Intersection-over-union threshold for NMS
        save       = True,             # Save annotated image to `runs/detect/...`
        save_txt   = True,             # Save raw YOLO-format text labels
        line_width = 1,                # Bounding-box line thickness
    )
    # After prediction, generate JSON summaries
    json: List[Any] = _generate_results_json(trained_dir_name, results)
    return results, json


def _generate_results_json(
    trained_dir_name: str,
    results: List[Results],
) -> list[Any]:
    """
    Convert the Ultralytics `Results` list into two JSON files:
    1. `results.json` with complete detection info (class id, name, confidence, coordinates).
    2. `report.json` with unique detected class names per image for reporting.

    Args:
        trained_dir_name (str): Name of directory under `data/reports/` to save JSON files.
        results (List[Results]): List of detection results from `predict()`.

    Returns:
        List[Any]: Report JSON Metadata.
    """
    # Prepare containers for full detections and summary (unique classes)
    full_detections = []
    summary_detections = []
    seen = set()
    for result in results:
        # Entry for full detection details
        entry_full = {"path": result.path, "boxes": []}
        # Entry for summary report (unique class names)
        entry_summary = {"path": result.path, "boxes": []}
        for box in result.boxes:
            cls_id = int(box.cls)
            name = result.names[cls_id]
            # Append full detection record
            entry_full["boxes"].append({
                "class"      : cls_id,
                "name"       : name,
                "confidence" : float(box.conf),
                "coordinates": [float(x) for x in box.xyxy[0].tolist()]
            })
            # For summary, only add each class once per image
            if name not in seen:
                seen.add(name)
                entry_summary["boxes"].append({
                    "class"      : cls_id,
                    "name"       : name,
                    "confidence" : float(box.conf),
                    "coordinates": [float(x) for x in box.xyxy[0].tolist()]
                })

        full_detections.append(entry_full)
        summary_detections.append(entry_summary)

    # Define output directory under data/reports/
    reports_path = os.path.join("data", "reports", trained_dir_name)
    os.makedirs(reports_path, exist_ok = True)

    # Save the comprehensive results JSON
    results_json_path = os.path.join(reports_path, "results.json")
    # 4) Salva o JSON
    with open(results_json_path, "w", encoding="utf-8") as file:
        json.dump(full_detections, file, indent = 2)
    print(f"✅ Detailed JSON saved to {results_json_path}")

    # Save the summary report JSON
    summary_json_path = os.path.join(reports_path, "report.json")
    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(summary_detections, file, indent = 2)
    print(f"✅ Summary JSON saved to {summary_json_path}")
    return full_detections


if __name__ == '__main__':
    trained_dir_name: str = 'yolo11n_custom_100'
    trained_model_best_path = f"../runs/detect/{trained_dir_name}/weights/best.pt"
    #source_file_path: str = "./data/sample/aws_01.jpg"
    #source_file_path: str = "./data/sample/aws_02.png"
    source_file_path: str = "./data/sample/aws_58.png"
    conf = 0.5

    results = predict(
        trained_dir_name,
        trained_model_best_path,
        source_file_path,
        conf
    )
