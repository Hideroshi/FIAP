import os
from collections import defaultdict

import cv2
from ultralytics import YOLO
from tqdm import tqdm
from deepface import DeepFace
import concurrent.futures

from image import draw_identified_box_ltrb
from text import save_results_to_csv, ensure_file_exists, \
                 count_emotion_appearances

# Função para analisar emoções com DeepFace
def analisar_emocao(face_img):
    try:
        face_resized = cv2.resize(face_img, (224, 224))
        resultado = DeepFace.analyze(
            face_resized,
            actions=['emotion'],
            detector_backend='mtcnn',
            enforce_detection=False
        )
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
    models_folder = os.path.join(base_dir, './doc/model')

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model_name = os.path.join(models_folder, "yolov11l-face.pt") # n, s, m, l, x versões disponiveis

    # Load de modelo pre treinado
    model = YOLO(model_name)  

    frame_idx = 0
    boxes_emocoes_prev = []
    results = defaultdict(list)
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
            emocoes = []
            for (x1, y1, x2, y2), emocao in boxes_emocoes_prev:
                emocoes.append(emocao)
                label = emocao if emocao else "Face"
                draw_identified_box_ltrb(frame, label, x1, y1, x2, y2)

            if boxes_emocoes_prev:
                results[frame_idx].extend(emocoes)
            else:
                results[frame_idx].append('')

            # Escrever o frame processado no vídeo de saída
            out.write(frame)
            frame_idx += 1
            cv2.imshow('Face Expression', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    out.release()
    save_results_to_csv(results, output_path + ".csv",
                        ("frame_id", "emotions_1", "emotions_2", "emotions_3",
                         "emotions_4"))
    write_summary_analysis(output_path)

def write_summary_analysis(
    video_out_path: str,
    analysis_output_path: str = "./doc/videos/result/summary_analysis.txt"
) -> None:
    ensure_file_exists(analysis_output_path)
    with open(analysis_output_path, "a", encoding="utf-8") as f:
        print(
            "\n    ====================== VIDEO SUMMARY: EMOTIONS (YOLO) =======================",
            file=f)
        counts = count_emotion_appearances(
            video_out_path + ".csv",
            pause_threshold=10,
            min_segment_length=5
        )
        for emotion, num in counts.items():
            print(f"    - {emotion}: {num} appearances", file=f)


def facial_expressions_with_yolo():
    # Caminho para o arquivo de vídeo na mesma pasta do script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, './doc/videos/tc4_video.mp4')  # Substitua 'video.mp4' pelo nome do seu vídeo
    output_path = os.path.join(base_dir, './doc/videos/result/tc4_video_fe_yolo.mp4')  # Nome do vídeo de saída

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
    facial_expressions_with_yolo()