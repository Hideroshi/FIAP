import os
import shutil
import random
import cv2
import numpy as np
from collections import defaultdict, Counter

# --- Configuração de caminhos e parâmetros ---
BASE_DIR     = 'data/dataset/aws'
SPLITS       = ['train', 'valid', 'test']
MIN_TRAIN    = 70      # meta de imagens por classe em train
MIN_VALID    = 15      # mínimo de imagens por classe em valid
MIN_TEST     = 15      # mínimo de imagens por classe em test
# -------------------------------------------

IMG_EXTS = ['.jpg', '.jpeg', '.png']
paths = {
    split: {
        'images': os.path.join(BASE_DIR, split, 'images'),
        'labels': os.path.join(BASE_DIR, split, 'labels')
    }
    for split in SPLITS
}

def load_class_files(split):
    cnt = Counter()
    files = defaultdict(set)
    lbl_dir = paths[split]['labels']
    for fn in os.listdir(lbl_dir):
        if not fn.endswith('.txt'): continue
        base = os.path.splitext(fn)[0]
        for line in open(os.path.join(lbl_dir, fn)):
            cls = int(line.split()[0])
            cnt[cls] += 1
            files[cls].add(base)
    return cnt, files

def find_image(base, split):
    img_dir = paths[split]['images']
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, base + ext)
        if os.path.exists(p): return p
    return None

def move_pair(base, src, dst):
    img = find_image(base, src)
    if img:
        shutil.move(img, os.path.join(paths[dst]['images'], os.path.basename(img)))
        print(f"    Moved IMAGE  {base} from {src} → {dst}")
    lbl = os.path.join(paths[src]['labels'], base + '.txt')
    if os.path.exists(lbl):
        shutil.move(lbl, os.path.join(paths[dst]['labels'], base + '.txt'))
        print(f"    Moved LABEL  {base} from {src} → {dst}")

def photometric_augment(img):
    alpha = random.uniform(0.8, 1.2)
    beta  = random.uniform(-20, 20)
    aug   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    noise = np.random.normal(0, 10, aug.shape).astype(np.uint8)
    return cv2.add(aug, noise)

if __name__ == '__main__':
    random.seed(42)

    # 1) Carrega contagens iniciais
    split_counts = {}
    split_files  = {}
    for s in SPLITS:
        c, f = load_class_files(s)
        split_counts[s] = c
        split_files[s]  = f

    oversampled  = 0
    moved_valid  = 0
    moved_test   = 0

    all_classes = set().union(*[c.keys() for c in split_counts.values()])

    # === FASE 1: OVERSAMPLING PARA TRAIN ===
    print("\n=== Phase 1: Oversampling → TRAIN ===")
    for cls in sorted(all_classes):
        current = split_counts['train'][cls]
        need    = max(0, MIN_TRAIN - current)
        print(f"\nClass {cls:3d}: train has {current}, target {MIN_TRAIN} → need {need}")
        if need == 0:
            print("  → OK, não precisa oversample")
            continue

        # escolhe fonte preferencial
        for src in ['train','valid','test']:
            if split_counts[src][cls] > 0:
                print(f"  → usando fonte {src} para oversample")
                bases = list(split_files[src][cls])
                break
        else:
            print(f"  ⚠️ sem origem para oversample da classe {cls}")
            continue

        random.shuffle(bases)
        for i in range(need):
            base = bases[i % len(bases)]
            img_path = find_image(base, src)
            lbl_path = os.path.join(paths[src]['labels'], base + '.txt')
            if not img_path or not os.path.exists(lbl_path):
                print(f"    ⚠️ pulando base {base} (arquivo não encontrado)")
                continue
            img = cv2.imread(img_path)
            aug = photometric_augment(img)
            new_base = f"{base}_os_{i}"
            ext = os.path.splitext(img_path)[1]
            cv2.imwrite(os.path.join(paths['train']['images'], new_base + ext), aug)
            shutil.copy(lbl_path, os.path.join(paths['train']['labels'], new_base + '.txt'))

            split_counts['train'][cls] += 1
            split_files['train'][cls].add(new_base)
            oversampled += 1
            #print(f"    ✔️ oversampled {new_base}")

    # === FASE 2: REBALANCE VALID ===
    print("\n=== Phase 2: Rebalance → VALID ===")
    for cls in sorted(all_classes):
        need = max(0, MIN_VALID - split_counts['valid'][cls])
        print(f"\nClass {cls:3d}: valid has {split_counts['valid'][cls]}, min {MIN_VALID} → need {need}")
        if need == 0:
            print("  → OK, não precisa rebalance valid")
            continue

        for _ in range(need):
            if split_counts['train'][cls] > 0:
                src = 'train'
            elif split_counts['test'][cls] > 0:
                src = 'test'
            else:
                print(f"  ⚠️ sem origem para rebalance valid classe {cls}")
                break
            print(f"  → movendo de {src} para valid")
            base = split_files[src][cls].pop()
            move_pair(base, src, 'valid')
            split_counts['valid'][cls] += 1
            split_counts[src][cls]  -= 1
            split_files['valid'][cls].add(base)
            moved_valid += 1

    # === FASE 3: REBALANCE TEST ===
    print("\n=== Phase 3: Rebalance → TEST ===")
    for cls in sorted(all_classes):
        cntt = split_counts['test'][cls]
        need = max(0, MIN_TEST - cntt)
        print(
            f"\nClass {cls:3d}: test has {cntt}, min {MIN_TEST} → need {need}")
        if need == 0:
            print("  → OK, não precisa rebalance test")
            continue

        for _ in range(need):
            # escolhe fonte: train → valid
            if split_counts['train'][cls] > 0 and split_files['train'][cls]:
                src = 'train'
            elif split_counts['valid'][cls] > 0 and split_files['valid'][cls]:
                src = 'valid'
            else:
                print(
                    f"  ⚠️ sem origem disponível para rebalance test da classe {cls}")
                break

            avail = list(split_files[src][cls])
            if not avail:
                print(
                    f"  ⚠️ não há arquivos restantes em '{src}' para classe {cls}")
                break

            # escolhe um base para mover (pode ser random.choice(avail) se preferir aleatório)
            base = avail[0]
            print(f"  → movendo '{base}' de {src} para test")

            # faz o move
            move_pair(base, src, 'test')

            # atualiza contadores e conjuntos
            split_files[src][cls].remove(base)
            split_counts[src][cls] -= 1

            split_files['test'][cls].add(base)
            split_counts['test'][cls] += 1
            moved_test += 1

        # opcional: mostra estado final desta classe
        print(
            f"  → test agora tem {split_counts['test'][cls]} para classe {cls}")

    # === RELATÓRIO FINAL ===
    print("\n=== Summary ===")
    print(f"Oversampled → TRAIN : {oversampled}")
    print(f"Moved to VALID     : {moved_valid}")
    print(f"Moved to TEST      : {moved_test}")
