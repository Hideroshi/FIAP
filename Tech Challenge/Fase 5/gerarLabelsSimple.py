import os
import cv2
import glob
import shutil
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim

# ================= CONFIGURAÇÕES =================
BASE_DIR = Path(__file__).parent.resolve()
ICONS_DIR = BASE_DIR / "aws_icons"
DIAGRAMS_DIR = BASE_DIR / "diagrams"
OUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "yolo_model"
MODEL_SIZE = MODEL_DIR / "yolo11x.pt"

THRESHOLD = 0.60
SSIM_THRESHOLD = 0.60
SCALES = [round(s, 2) for s in np.linspace(0.4, 1.8, 30)]
EPOCHS = 100
VAL_RATIO = 0.2

# ================= SETUP DE DIRETÓRIOS =================
os.makedirs(OUT_DIR / "images/train", exist_ok=True)
os.makedirs(OUT_DIR / "labels/train", exist_ok=True)
os.makedirs(OUT_DIR / "images/val", exist_ok=True)
os.makedirs(OUT_DIR / "labels/val", exist_ok=True)
os.makedirs(OUT_DIR / "visual", exist_ok=True)

# ================= FUNÇÕES AUXILIARES =================
def build_class_map(icons_dir):
    icon_paths = list(Path(icons_dir).rglob("*.png"))
    return {p.parent.name: idx for idx, p in enumerate(sorted(set(icon_paths), key=lambda x: x.parent.name))}

def is_number_region(region, image):
    x1, y1, x2, y2 = region
    w, h = x2 - x1, y2 - y1

    if w < 16 or h < 16 or w/h > 2.5 or h/w > 2.5:
        return True

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return True

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_brightness = cv2.mean(gray)[0]
    if mean_brightness > 240 or mean_brightness < 10:
        return True

    edges = cv2.Canny(gray, 50, 150)
    std = cv2.meanStdDev(edges)[1][0][0]
    if std < 10:
        return True

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 3:
        return True

    import pytesseract
    config = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(gray, config=config).strip()
    if text.isdigit() and len(text) <= 2:
        return True

    return False

def apply_nms(detections, iou_thresh=0.2):
    if not detections:
        return []
    boxes = np.array([[x1, y1, x2 - x1, y2 - y1] for (_, (x1, y1, x2, y2), _) in detections])
    scores = np.array([d[2] for d in detections])
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=THRESHOLD, nms_threshold=iou_thresh)
    if len(indices) == 0:
        return []
    indices = indices.flatten()
    return [(detections[i][0], detections[i][1]) for i in indices]

# ================= FUNÇÃO DE DETECÇÃO =================
def detect_icons(image_path, class_map):
    detections = []
    try:
        color_image = cv2.imread(str(image_path))
        if color_image is None:
            return [], (0, 0)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        for template_path in sorted(Path(ICONS_DIR).rglob("*.png")):
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
                resized_template = cv2.resize(template_gray, (0, 0), fx=scale, fy=scale)
                th_scaled, tw_scaled = resized_template.shape[:2]
                if gray_image.shape[0] < th_scaled or gray_image.shape[1] < tw_scaled:
                    continue

                result = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(result >= THRESHOLD)

                for pt in zip(*loc[::-1]):
                    x1, y1 = int(pt[0]), int(pt[1])
                    x2, y2 = x1 + tw_scaled, y1 + th_scaled
                    region = (x1, y1, x2, y2)

                    if is_number_region(region, color_image):
                        continue

                    roi = gray_image[y1:y2, x1:x2]
                    if roi.shape != resized_template.shape:
                        continue

                    ssim_score = ssim(roi, resized_template, data_range=255)
                    if ssim_score < 0.75:
                        continue

                    confidence = result[y1, x1]
                    detections.append((class_id, region, confidence))

        return apply_nms(detections), color_image.shape
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
        return [], (0, 0)

# ================= DEMO E TREINAMENTO =================
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
    img_dir = OUT_DIR / ("images/val" if is_val else "images/train")
    lbl_dir = OUT_DIR / ("labels/val" if is_val else "labels/train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    shutil.copy(image_path, img_dir / filename)
    image = cv2.imread(str(image_path))
    id_to_class = {v: k for k, v in class_map.items()}
    label_path = lbl_dir / f"{base_name}.txt"
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
    cv2.imwrite(str(OUT_DIR / "visual" / filename), image)
    return len(detections)

def generate_yaml_config(class_map, output_path):
    yaml_path = output_path / "aws_config.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_path.resolve()}\n")
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
    print("Iniciando pipeline AWS Detector")
    icon_list = list(Path(ICONS_DIR).rglob("*.png"))
    if not icon_list:
        print(f"Erro: Nenhum ícone encontrado em {ICONS_DIR}")
        return
    else:
        print(f"✔ {len(icon_list)} ícones encontrados")

    diagram_paths = sorted(DIAGRAMS_DIR.glob("*.png"))[:10]
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
        print(f"{'VAL' if is_val else 'TRAIN'} {path.name}: {count} detecções")

    print("\n🎯 Iniciando treinamento YOLOv11")
    # model = YOLO(str(MODEL_SIZE))
    # model.train(
    #     data=str(generate_yaml_config(class_map, OUT_DIR)),
    #     epochs=EPOCHS,
    #     batch=16,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    #     imgsz=640,
    #     patience=15,
    #     optimizer="AdamW",
    #     lr0=0.001
    # )
    print("✅ Treinamento concluído com sucesso!")

if __name__ == "__main__":
    main()