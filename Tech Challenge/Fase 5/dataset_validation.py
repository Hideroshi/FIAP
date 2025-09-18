# validate_dataset.py

import os
import glob
import yaml
from collections import Counter, defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
base_dir = "data/dataset/aws"     # ajuste para a raiz do seu aws/
splits   = ["train", "valid", "test"]
# ────────────────────────────────────────────────────────────────────────────────

# 1) Carrega data.yaml
cfg   = yaml.safe_load(open(os.path.join(base_dir, "data.yaml")))
nc    = cfg["nc"]
names = cfg["names"]

# 1.a) Conta imagens e labels (base names)
images = set()
labels = set()
for split in splits:
    img_dir = os.path.join(base_dir, split, "images")
    lbl_dir = os.path.join(base_dir, split, "labels")
    for p in glob.glob(os.path.join(img_dir, "*")):
        images.add(os.path.splitext(os.path.basename(p))[0])
    for p in glob.glob(os.path.join(lbl_dir, "*.txt")):
        labels.add(os.path.splitext(os.path.basename(p))[0])

len_names  = len(names)
len_images = len(images)
len_labels = len(labels)

print("── Passo 1: contagens ───────────────────────────────────────────────────────")
print(f"nc declarado:       {nc}")
print(f"len(names):         {len_names}")
print(f"total imagens:      {len_images}")
print(f"total labels:       {len_labels}")

if nc != len_names:
    print(f"❌ ERRO: nc ({nc}) != len(names) ({len_names})")
else:
    print("✅ nc == len(names)")

if len_images != len_labels:
    print(f"❌ ERRO: #imagens ({len_images}) != #labels ({len_labels})")
else:
    print("✅ #imagens == #labels")

# 2) Verifica duplicatas em names
print("\n── Passo 2: duplicatas em names ─────────────────────────────────────────────")
dupes = [name for name,count in Counter(names).items() if count>1]
if dupes:
    print("❌ Encontradas duplicatas:")
    for name in dupes:
        idxs = [i for i,n in enumerate(names) if n==name]
        print(f"   – '{name}' nos índices {idxs}")
else:
    print("✅ Nenhuma duplicata em names")

# 3) Consistência labels x names x imagens
print("\n── Passo 3: consistência labels x names x imagens ──────────────────────────")

# vamos mapear cada class_id para os arquivos onde aparece
files_by_id = defaultdict(list)
used_ids    = set()
invalid_ids = set()

for split in splits:
    lbl_dir = os.path.join(base_dir, split, "labels")
    if not os.path.isdir(lbl_dir):
        continue
    for txt in glob.glob(os.path.join(lbl_dir, "*.txt")):
        for line in open(txt):
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(parts[0])
            used_ids.add(cid)
            if cid < 0 or cid >= nc:
                invalid_ids.add(cid)
                files_by_id[cid].append(os.path.relpath(txt))

if invalid_ids:
    print("❌ Encontrados class_id em labels sem names correspondentes:")
    for cid in sorted(invalid_ids):
        print(f"\n   • class_id {cid} (não há names[{cid}])")
        print(f"     Aparece {len(files_by_id[cid])} vez(es) em:")
        for f in files_by_id[cid][:5]:
            print(f"       – {f}")
        if len(files_by_id[cid]) > 5:
            print(f"       …e mais {len(files_by_id[cid]) - 5} arquivos")
else:
    print("✅ Todos os class_id em labels têm names associados")

# 3.a) Imagens sem label e labels sem imagem
only_imgs = sorted(images - labels)
only_lbls = sorted(labels - images)
if only_imgs:
    print(f"\n❌ {len(only_imgs)} imagens sem .txt correspondente: {only_imgs[:5]} ...")
else:
    print("\n✅ Todas as imagens têm .txt correspondente")

if only_lbls:
    print(f"❌ {len(only_lbls)} labels sem imagem correspondente: {only_lbls[:5]} ...")
else:
    print("✅ Todos os labels têm imagem correspondente")

# 3.b) Classes em names sem exemplo nos labels
unused = sorted(set(range(nc)) - used_ids)
if unused:
    print(f"\n❌ {len(unused)} classes em names SEM exemplos nos labels:")
    for cid in unused[:10]:
        print(f"   – {cid}: {names[cid]}")
    if len(unused) > 10:
        print(f"   …e mais {len(unused) - 10} classes")
else:
    print("\n✅ Todas as classes em names têm pelo menos 1 exemplo")




