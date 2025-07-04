from ultralytics import YOLO
import cv2

# Detecção com YOLOv8
def detect_icons(image_path, model_path='yolo_model/best.pt'):
    model = YOLO(model_path)
    results = model(image_path)[0]
    
    detections = []
    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls_id = r.tolist()
        detections.append({
            'label': model.names[int(cls_id)],
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'confidence': conf
        })
    return detections

# OCR com pytesseract
import pytesseract

def extract_text_from_region(image, bbox):
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    text = pytesseract.image_to_string(roi)
    return text.strip()

import networkx as nx


# Montar grafo com NetworkX
def build_graph(detections, image):
    G = nx.DiGraph()
    for idx, det in enumerate(detections):
        label = det['label']
        text = extract_text_from_region(image, det['bbox'])
        node_id = f"{label}_{idx}"
        G.add_node(node_id, label=label, text=text, pos=det['bbox'])
    
    # Exemplo básico de ligação: conecta todos com todos (você pode usar heurísticas com distância)
    nodes = list(G.nodes(data=True))
    for i, (n1, d1) in enumerate(nodes):
        for j, (n2, d2) in enumerate(nodes):
            if i != j:
                G.add_edge(n1, n2)
    
    return G

# Visualizar o grafo
import matplotlib.pyplot as plt

def visualize_graph(G):
    pos = {n: ((d['pos'][0]+d['pos'][2])//2, -(d['pos'][1]+d['pos'][3])//2) for n, d in G.nodes(data=True)}
    labels = {n: f"{d['label']}" for n, d in G.nodes(data=True)}
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=1500, node_color='lightblue')
    plt.show()

# Gerar código com diagrams

from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2
from diagrams.aws.storage import S3
from diagrams.aws.database import RDS

def generate_diagram(G):
    with Diagram("Generated Cloud Diagram", show=True):
        nodes = {}
        for n, d in G.nodes(data=True):
            label = d['label'].lower()
            if 'ec2' in label:
                nodes[n] = EC2(d['text'] or 'EC2')
            elif 's3' in label:
                nodes[n] = S3(d['text'] or 'S3')
            elif 'rds' in label:
                nodes[n] = RDS(d['text'] or 'RDS')
            else:
                nodes[n] = EC2(d['text'] or 'Generic')

        for src, dst in G.edges:
            nodes[src] >> nodes[dst]


def main():
    image_path = "diagram.png"
    image = cv2.imread(image_path)

    detections = detect_icons(image_path)
    graph = build_graph(detections, image)
    visualize_graph(graph)  # opcional
    generate_diagram(graph)

if __name__ == "__main__":
    main()