import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from deepface import DeepFace
import concurrent.futures
 
# Função para analisar emoções com DeepFace
def analisar_emocao(face_img):
    try:
        face_resized = cv2.resize(face_img, (224, 224))
        resultado = DeepFace.analyze(face_resized, actions=['emotion'], detector_backend='dlib', enforce_detection=False)
        return resultado[0]['dominant_emotion']
    except Exception as e:
        print(f"Erro ao detectar emoção: {e}")
        return None

# Analisar o frame para detectar faces e expressões
def detectar_pessoas(frame, model, processar_emocao=False, executor=None):
    boxes_emocoes = []
    results = model.predict(frame)

    # Desenha as caixas e os labels de pessoas no frame
    for result in results:
        if len(result):
            face_imgs = []
            coords = []
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)  # Garante que os índices são válidos
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                face_img = frame[y1:y2, x1:x2]
                
                if processar_emocao:
                    face_imgs.append(face_img)
                    coords.append((x1, y1, x2, y2))
                else:
                    # Apenas manter coords com emoção vazia (será substituído depois)
                    boxes_emocoes.append(((x1, y1, x2, y2), None))
                 

            # Se for pra analisar emoções, fazer em paralelo
            if processar_emocao and face_imgs:
                emocoes = list(executor.map(analisar_emocao, face_imgs))
                for coord, emocao in zip(coords, emocoes):
                    boxes_emocoes.append((coord, emocao))

    return boxes_emocoes

def processar_video(cap, output_path):
    # Caminho para a pasta de imagens com rostos conhecidos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(base_dir, 'images')
    models_folder = os.path.join(base_dir, 'models')

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model_name = os.path.join(models_folder, "yolov11l-face.pt")

    # Load de modelo pre treinado
    model = YOLO(model_name)  

    frame_idx = 0
    boxes_emocoes_prev = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Loop para processar cada frame do vídeo com barra de progresso
        for _ in tqdm(range(total_frames), desc="Processando vídeo"):
            # Ler um frame do vídeo
            ret, frame = cap.read()

            # Se não conseguiu ler o frame (final do vídeo), sair do loop
            if not ret:
                break

            if frame_idx % 5 == 0:  # a cada 5 frames
                boxes_emocoes_prev = detectar_pessoas(frame, model, processar_emocao=True, executor=executor)
           
            # Apenas reutilizar boxes anteriores
            for (x1, y1, x2, y2), emocao in boxes_emocoes_prev:
                label = emocao if emocao else "Face"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1+6, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    
            # Escrever o frame processado no vídeo de saída
            out.write(frame)
            frame_idx += 1

    out.release()

def main_loop():
    # Caminho para o arquivo de vídeo na mesma pasta do script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, 'video.mp4')  # Substitua 'video.mp4' pelo nome do seu vídeo
    output_path = os.path.join(base_dir, 'output_video.mp4')  # Nome do vídeo de saída

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)  
    
    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    processar_video(cap, output_path)

    # Liberar o vídeo e fecha todas as janelas
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    main_loop()