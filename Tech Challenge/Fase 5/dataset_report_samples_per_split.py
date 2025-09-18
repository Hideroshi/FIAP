import os
from pathlib import Path
from collections import Counter

from dataset_report_low_counts import get_data_yaml

"""
This script loads class names from a YOLO data.yaml file and counts
how many label occurrences each class has in the train, valid, and test splits.
It outputs a formatted table listing each class name, its ID, and the counts per split.

Workflow:
1. Define `count_folder` to scan a labels directory and tally class IDs.
2. Invoke `count_folder` for train/valid/test label directories.
3. Load `names` list from data.yaml to map class IDs to human-readable names.
4. Print a table header and iterate over each class ID present in training,
   displaying counts from all splits (0 if missing).

Usage:
    python dataset_report_samples_per_split.py

Requirements:
- A YOLO-format `data.yaml` with a `names` entry adjacent to the split folders.
- Directory structure:
    data/dataset/aws/
        data.yaml
        train/labels/*.txt
        valid/labels/*.txt
        test/labels/*.txt
"""

def count_folder(label_dir: str) -> Counter:
    """
    Scan a directory of YOLO .txt label files and count occurrences per class ID.

    Args:
        label_dir (str): Path to a folder containing .txt label files.

    Returns:
        Counter: Mapping from class_id (int) to total count of label lines.
    """
    cnt = Counter()
    for fn in os.listdir(label_dir):
        if fn.endswith('.txt'):
            for line in open(os.path.join(label_dir, fn)):
                cls = int(line.split()[0])
                cnt[cls] += 1
    return cnt

# ------- Configuration -------
data_yaml_path = Path('data/dataset/aws/data.yaml')
train_cnt = count_folder('data/dataset/aws/train/labels')
valid_cnt = count_folder('data/dataset/aws/valid/labels')
test_cnt = count_folder('data/dataset/aws/test/labels')
# -----------------------------

names = get_data_yaml('data/dataset/aws/data.yaml')['names']

print('Classe                     |    id | Train | Valid | Test')
for cls in sorted(train_cnt):
    print(f'{names[cls]:26s} | {cls:5d} | {train_cnt[cls]:5d} | {valid_cnt.get(cls,0):5d} | {test_cnt.get(cls,0):4d}')

