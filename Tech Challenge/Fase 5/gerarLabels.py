import os
import cv2
import glob
import shutil
from pathlib import Path
from ultralytics import YOLO

# CONFIGURAÇÃO
ICONS_DIR = "aws_icons"
DIAGRAMS_DIR = "aws_dataset_raw"
OUT_DIR = "output"
OUT_IMG_DIR = os.path.join(OUT_DIR, "images/train")
OUT_LBL_DIR = os.path.join(OUT_DIR, "labels/train")
THRESHOLD = 0.75
EPOCHS = 50
IMG_SIZE = 640
MODEL_SIZE = "yolov8s.pt"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# Mapeia classes com base na pasta e nome do ícone
def build_class_map(icons_dir):
    icon_paths = list(Path(icons_dir).rglob("*.png"))
    class_names = sorted(set(p.parent.name + "_" + p.stem for p in icon_paths))
    return {name: idx for idx, name in enumerate(class_names)}

def detect_icons(image_path, class_map):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = []

    for template_path in Path(ICONS_DIR).rglob("*.png"):
        template = cv2.imread(str(template_path), 0)
        if template is None or template.shape[0] < 10 or template.shape[1] < 10:
            continue

        template_name = template_path.parent.name + "_" + template_path.stem
        class_id = class_map.get(template_name)
        if class_id is None:
            continue

        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        loc = (res >= THRESHOLD).nonzero()

        for pt in zip(*loc[::-1]):
            x1, y1 = pt
            x2, y2 = x1 + w, y1 + h
            detections.append((class_id, (x1, y1, x2, y2)))

    return detections, image.shape

def convert_to_yolo_format(bbox, image_shape):
    x1, y1, x2, y2 = bbox
    img_h, img_w = image_shape[:2]
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height

def save_detection(image_path, detections, image_shape):
    filename = os.path.basename(image_path)
    label_file = filename.rsplit(".", 1)[0] + ".txt"

    shutil.copy(image_path, os.path.join(OUT_IMG_DIR, filename))

    with open(os.path.join(OUT_LBL_DIR, label_file), "w") as f:
        for class_id, bbox in detections:
            x, y, w, h = convert_to_yolo_format(bbox, image_shape)
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def generate_yaml_config(class_map, output_path):
    yaml_path = os.path.join(output_path, "cloud.yaml")
    with open(yaml_path, "w") as f:
        f.write("path: " + os.path.abspath(output_path) + "\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")  # você pode ajustar depois para usar uma pasta val
        f.write(f"nc: {len(class_map)}\n")
        f.write("names: [\n")
        for name in sorted(class_map, key=lambda x: class_map[x]):
            f.write(f"  '{name}',\n")
        f.write("]\n")
    return yaml_path

def main():
    print("🔍 Gerando dataset anotado...")
    class_map = build_class_map(ICONS_DIR)
    for diagram_path in glob.glob(os.path.join(DIAGRAMS_DIR, "*.png")):
        detections, shape = detect_icons(diagram_path, class_map)
        save_detection(diagram_path, detections, shape)
        print(f"✓ {os.path.basename(diagram_path)} - {len(detections)} detecções.")

    print("\n📝 Gerando arquivo YAML de configuração...")
    yaml_path = generate_yaml_config(class_map, OUT_DIR)
    print(f"✓ YAML salvo em: {yaml_path}")

    print("\n🚀 Iniciando treinamento YOLOv8...")
    model = YOLO(MODEL_SIZE)
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=16,
        project="runs/train",
        name="cloud_architecture",
    )
    print("✅ Treinamento concluído!")

if __name__ == "__main__":
    main()
