import os
from collections import Counter
from pathlib import Path

from dataset_report_low_counts import get_data_yaml

"""
This script loads class names from a YOLO-formatted data.yaml file and
counts label occurrences in train, test, and valid splits. Classes with
low representation (below configurable thresholds) are identified,
and a formatted report is printed showing counts per split.

Configuration:
- data_yaml_path: Path to the dataset YAML defining splits and class names.
- train_label_path, test_label_path, valid_label_path: Paths to .txt label dirs.
- min_train: Minimum required training examples per class.
- min_valid: Minimum required validation examples per class.

Usage:
    python dataset_report_missing_splits.py
"""
def load_classes(data_yaml_path: Path) -> list[str]:
    """
    Load class names from a YOLO-format data.yaml file.

    Args:
        data_yaml_path (str): Path to the data.yaml containing 'names' or 'classes'.

    Returns:
        list[str]: Ordered list of class names.

    Raises:
        ValueError: If  'names' is found in the YAML.
    """
    data = get_data_yaml(data_yaml_path)
    names = data["names"]
    if not names:
        raise ValueError("Could not find 'names' in data.yaml")
    return names


def count_labels(split_dir: str) -> Counter:
    """
    Count occurrences of each class ID in the labels folder of a given split.

    Args:
        split_dir (str): Base directory of a split containing a 'labels' subfolder.

    Returns:
        Counter: Mapping from class_id (int) to count of label lines.
    """
    cnt = Counter()
    labels_dir = os.path.join(split_dir, "labels")
    if not os.path.isdir(labels_dir):
        return cnt
    for fn in os.listdir(labels_dir):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, fn), "r") as f:
            for line in f:
                cls = int(line.split()[0])
                cnt[cls] += 1
    return cnt


def get_split_counts( base_dir: Path, splits: list[str]) -> dict[str, Counter]:
    """
    Aggregate label counts for each split directory.

    Args:
        base_dir (Path): Root directory containing split subfolders.
        splits (list[str]): List of split names (e.g., ['train','valid','test']).

    Returns:
        dict[str, Counter]: Mapping split name -> Counter of class occurrences.
    """
    split_counts = {}
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        split_counts[split] = count_labels(split_dir)
    return split_counts


def print_missing(class_names: list[str], split_counts: dict[str, Counter]) -> None:
    """
    Print classes that are missing in one or more splits.

    Args:
        class_names (list[str]): Ordered list of class names.
        split_counts (dict[str, Counter]): Mapping of split names to label counts.
    """
    any_missing = False
    for cls_idx, cls_name in enumerate(class_names):
        missing = [split for split, cnt in split_counts.items() if cnt.get(cls_idx, 0) == 0]
        if missing:
            any_missing = True
            print(f"Class {cls_idx:3d} ('{cls_name}') missing in: {', '.join(missing)}")
    if not any_missing:
        print("✅ Every class has at least one label in each of train, valid, and test.")


if __name__ == "__main__":
    # ------- Configuration -------
    base_dir = Path('data/dataset/aws/')
    data_yaml_path = Path('data/dataset/aws/data.yaml')
    splits = ["train", "valid", "test"]
    # -----------------------------

    # Load class names
    class_names = load_classes(data_yaml_path)

    # For each split, get a Counter of class→count
    split_counts = get_split_counts(base_dir, splits)

    # For each class, see which splits have zero
    print_missing(class_names, split_counts)