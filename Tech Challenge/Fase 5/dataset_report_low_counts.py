import yaml
from collections import Counter
from pathlib import Path

"""
This script loads class names from a YOLO data.yaml file and counts
label occurrences in train, test, and valid splits. It then identifies
classes with low representation based on configurable thresholds and
prints a formatted report showing how many instances each low-count
class has per split.

Configuration:
- data_yaml: path to the dataset YAML file defining splits and class names.
- train_dir, test_dir, valid_dir: paths to label directories for each split.
- min_train: minimum number of train examples required to be considered sufficient.
- min_valid: minimum number of valid examples required.

Usage:
    python dataset_report_low_counts.py
"""
def count_labels(folder: Path) -> Counter:
    """
    Count occurrences of each class ID across all .txt label files in folder.

    Args:
        folder (Path): Directory of YOLO-format .txt label files.

    Returns:
        Counter: class_id -> occurrence count.
    """
    cnt = Counter()
    for label_file in folder.glob('*.txt'):
        with open(label_file) as lf:
            for line in lf:
                parts = line.split()
                if not parts:
                    continue
                class_id = int(parts[0])
                cnt[class_id] += 1
    return cnt


def get_data_yaml(data_yaml_path: Path) -> list[str]:
    """
    Load data.yaml file.

    Args:
        data_yaml_path (Path): Path to the YAML file containing 'names'.

    Returns:
        List[str]: Class names, index corresponds to class ID.
    """
    with open(data_yaml_path) as file:
        data = yaml.safe_load(file)
    return data


def get_low_classes(
    data_yaml_path: Path,
    train_label_path: Path,
    test_label_path: Path,
    valid_label_path: Path,
    min_train: int,
    min_valid: int,
) -> list[tuple[int, str, int, int, int]]:
    """
    Identify classes with counts below thresholds in train or valid splits.

    Args:
        data_yaml_path (Path): Dataset YAML file path.
        train_label_path (Path): Directory of training .txt labels.
        test_label_path (Path): Directory of test .txt labels.
        valid_label_path (Path): Directory of valid .txt labels.
        min_train (int): Minimum required train count.
        min_valid (int): Minimum required valid count.

    Returns:
        List of tuples (class_id, class_name, train_count, test_count, valid_count).
    """
    class_names = get_data_yaml(data_yaml_path)['names']
    train_counts = count_labels(train_label_path)
    test_counts = count_labels(test_label_path)
    valid_counts = count_labels(valid_label_path)

    low_classes  = []
    for idx, name in enumerate(class_names):
        t = train_counts.get(idx, 0)
        e = test_counts.get(idx, 0)
        v = valid_counts.get(idx, 0)
        if t < min_train or v < min_valid:
            low_classes.append((idx, name, t, e, v))
    return low_classes


if __name__ == '__main__':
    # --- Configuration ---
    data_yaml_path = Path('data/dataset/aws/data.yaml')
    train_label_path = Path('data/dataset/aws/train/labels')
    test_label_path = Path('data/dataset/aws/test/labels')
    valid_label_path = Path('data/dataset/aws/valid/labels')

    min_train = 20  # minimum items in train to be sufficient
    min_valid = 5  # minimum items in valid to be sufficient
    # ----------------------

    low_classes = get_low_classes(
        data_yaml_path,
        train_label_path,
        test_label_path,
        valid_label_path,
        min_train,
        min_valid
    )
    header = f"{'class':26} | {'idx':>3} | {'train':>5} | {'test':>5} | {'valid':>5}"
    print(header)
    print('-' * len(header))
    for idx, name, t, e, v in low_classes:
        print(f"{name:26} | {idx:3d} | {t:5d} | {e:5d} | {v:5d}")
