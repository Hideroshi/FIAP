import os
import cv2
import glob
import shutil
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ================= CONFIGURAÇÕES =================
ICONS_DIR = "aws_icons"
DIAGRAMS_DIR = "diagrams"
OUT_DIR = "output"
MODEL_DIR = "yolo_model"
MODEL_SIZE = os.path.join(MODEL_DIR, "yolo11x.pt")

THRESHOLD = 0.63
SCALES = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
EPOCHS = 50
VAL_RATIO = 0.2

# ================= SETUP DE DIRETÓRIOS =================
os.makedirs(os.path.join(OUT_DIR, "images/train"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "images/val"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "labels/val"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "visual"), exist_ok=True)

# ================= FUNÇÕES PRINCIPAIS =================

def build_class_map(icons_dir):
    icon_paths = list(Path(icons_dir).rglob("*.png"))
    return {p.parent.name: idx for idx, p in enumerate(sorted(set(icon_paths), key=lambda x: x.parent.name))}

def is_number_region(region, image):
    x1, y1, x2, y2 = region
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if cv2.mean(gray)[0] > 200:
        return True
    edges = cv2.Canny(gray, 50, 150)
    std = cv2.meanStdDev(edges)[1][0][0]
    if std < 15:
        return True
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return True
    return False

def iou(box1, box2):
    xi = max(box1[0], box2[0])
    yi = max(box1[1], box2[1])
    xu = min(box1[2], box2[2])
    yu = min(box1[3], box2[3])
    inter = max(0, xu - xi) * max(0, yu - yi)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def apply_nms(detections, iou_thresh=0.5):
    if not detections:
        return []
    detections.sort(key=lambda x: x[2], reverse=True)
    final = []
    while detections:
        current = detections.pop(0)
        keep = True
        for other in final:
            if current[0] == other[0] and iou(current[1], other[1]) > iou_thresh:
                keep = False
                break
        if keep:
            final.append((current[0], current[1]))
    return final

def refine_bbox(region, image):
    """Ajusta a bbox ao conteúdo real dentro da região"""
    x1, y1, x2, y2 = region
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return region  # fallback

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x1 + x, y1 + y, x1 + x + w, y1 + y + h)

def detect_icons(image_path, class_map):
    detections = []
    try:
        color_image = cv2.imread(image_path)
        if color_image is None:
            return [], (0, 0)
        grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        for template_path in Path(ICONS_DIR).rglob("*.png"):
            class_id = class_map.get(template_path.parent.name)
            if class_id is None:
                continue
            template_orig = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
            if template_orig is None:
                continue
            if template_orig.shape[2] == 4:
                alpha = template_orig[:, :, 3]
                _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
                template_rgb = cv2.bitwise_and(template_orig[:, :, :3], template_orig[:, :, :3], mask=mask)
            else:
                template_rgb = template_orig
            template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)
            for scale in SCALES:
                template = cv2.resize(template_gray, (0, 0), fx=scale, fy=scale)
                h, w = template.shape[:2]
                if h < 20 or w < 20 or h > grayscale_image.shape[0] or w > grayscale_image.shape[1]:
                    continue
                result = cv2.matchTemplate(grayscale_image, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(result >= THRESHOLD)
                for pt in zip(*loc[::-1]):
                    top_left = pt
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    raw_region = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
                    if is_number_region(raw_region, color_image):
                        continue
                    refined_region = refine_bbox(raw_region, color_image)
                    confidence = result[top_left[1], top_left[0]]
                    detections.append((class_id, refined_region, confidence))
        return apply_nms(detections), color_image.shape
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
        return [], (0, 0)

def convert_to_yolo_format(bbox, image_shape):
    x1, y1, x2, y2 = bbox
    img_w, img_h = image_shape[1], image_shape[0]
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height

def save_detection(image_path, detections, image_shape, class_map, is_val=False):
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    img_dir = os.path.join(OUT_DIR, "images/val" if is_val else "images/train")
    lbl_dir = os.path.join(OUT_DIR, "labels/val" if is_val else "labels/train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    shutil.copy(image_path, os.path.join(img_dir, filename))
    image = cv2.imread(image_path)
    id_to_class = {v: k for k, v in class_map.items()}
    label_path = os.path.join(lbl_dir, f"{base_name}.txt")
    with open(label_path, "w") as f:
        for class_id, bbox in detections:
            x, y, w, h = convert_to_yolo_format(bbox, image_shape)
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = id_to_class[class_id]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1-th-4), (x1+tw, y1), color, -1)
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    cv2.imwrite(os.path.join(OUT_DIR, "visual", filename), image)
    return len(detections)

def generate_yaml_config(class_map, output_path):
    yaml_path = os.path.join(output_path, "aws_config.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_path)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_map)}\n")
        f.write("names: [\n")
        for name in sorted(class_map.keys()):
            f.write(f"  '{name}',\n")
        f.write("]\n")
    return yaml_path

def split_diagrams(diagram_paths):
    np.random.shuffle(diagram_paths)
    split_idx = int(len(diagram_paths) * (1 - VAL_RATIO))
    return diagram_paths[:split_idx], diagram_paths[split_idx:]

def main():
    print("🚀 Iniciando pipeline AWS Detector")
    if not list(Path(ICONS_DIR).rglob("*.png")):
        print(f"Erro: Nenhum ícone encontrado em {ICONS_DIR}")
        return
    diagram_paths = sorted(glob.glob(os.path.join(DIAGRAMS_DIR, "*.png")))[:10]
    if not diagram_paths:
        print(f"Erro: Nenhum diagrama encontrado em {DIAGRAMS_DIR}")
        return
    class_map = build_class_map(ICONS_DIR)
    print(f"🔍 {len(class_map)} classes identificadas")
    train_paths, val_paths = split_diagrams(diagram_paths)
    print(f"📊 Dataset: {len(train_paths)} treino, {len(val_paths)} validação")
    for path in train_paths + val_paths:
        is_val = path in val_paths
        detections, shape = detect_icons(path, class_map)
        count = save_detection(path, detections, shape, class_map, is_val)
        print(f"{'VAL' if is_val else 'TRAIN'} {os.path.basename(path)}: {count} detecções")
    print("\n🎯 Iniciando treinamento YOLOv11")
    model = YOLO(MODEL_SIZE)
    model.train(
        data=generate_yaml_config(class_map, OUT_DIR),
        epochs=EPOCHS,
        batch=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        imgsz=640,
        patience=15,
        optimizer="AdamW",
        lr0=0.001
    )
    print("✅ Treinamento concluído com sucesso!")

if __name__ == "__main__":
    main()
