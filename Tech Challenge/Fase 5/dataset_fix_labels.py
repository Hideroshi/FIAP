from pathlib import Path
from typing import List
from PIL import Image
"""
This script detects and converts mixed segmentation (polygon) annotations
in YOLO label files to bounding-box format. It processes each .txt label
file in the specified labels directory, computes bounding boxes from
polygon coordinates if necessary, and writes corrected label files.

Functions:
- fix_labels(images_dir: str,
             labels_dir: str,
             output_dir: Optional[str] = None) -> None:
    Main routine to scan label files, convert polygon segments to
    YOLO detection format, and save results.

Usage Example:
    python dataset_fix_labels.py
"""
def fix_labels(
    images_dir: str,
    labels_dir: str,
    output_dir: str = None
) -> None:
    """
    Convert mixed segmentation annotations to YOLO detection labels.

    Scans all .txt files in labels_dir. Each line is expected to be:
      class_id x1 y1 x2 y2 ... xn yn
    - If a line has exactly 5 values, it is already in detection format
      (class_id x_center y_center width height) and is left unchanged.
    - If a line has more than 5 values, it is interpreted as polygon
      coordinates. The bounding box is computed as the min/max of x and y,
      then converted to normalized YOLO detection format.

    Args:
        images_dir (str): Directory containing image files (.jpg/.png).
        labels_dir (str): Directory containing YOLO-format .txt label files.
        output_dir (str, optional): If provided, corrected labels are
            written here. Otherwise original files are overwritten.

    Behavior:
        - Loads each corresponding image to determine width/height for
          pixel-to-normalized conversion if polygon coords > 1.
        - Writes corrected label file when at least one line was fixed.
        - Prints a message for each fixed file and a summary count.
    """
    labels_path = Path(labels_dir)
    imgs_path = Path(images_dir)
    out_path = Path(output_dir) if output_dir else labels_path
    out_path.mkdir(parents=True, exist_ok=True)

    txt_files = list(labels_path.glob('*.txt'))
    total_fixed = 0
    for txt_file in txt_files:
        img_file = imgs_path / (txt_file.stem + '.jpg')
        if not img_file.exists():
            img_file = imgs_path / (txt_file.stem + '.png')
        # If image exists, get size for absolute coords -> normalized conversion
        if img_file.exists():
            with Image.open(img_file) as img:
                width, height = img.size
        else:
            width = height = None  # assume coords normalized

        new_lines: List[str] = []
        fixed = False
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                cls = parts[0]
                coords = list(map(float, parts[1:]))
                # detection line: class + 4 values
                if len(parts) == 5:
                    new_lines.append(line.strip())
                    continue
                # segment line: class + >=6 coords
                xs = coords[0::2]
                ys = coords[1::2]
                # if image size known and coords appear >1, assume pixel coords and normalize
                if width and height and max(xs) > 1 or max(ys) > 1:
                    xs = [x/width for x in xs]
                    ys = [y/height for y in ys]
                # compute bounding box in normalized coords
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                w = x_max - x_min
                h = y_max - y_min
                new_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                fixed = True

        if fixed:
            total_fixed += 1
            out_file = out_path / txt_file.name
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(new_lines) + "\n")
            print(f"Fixed segments in: {txt_file.name}")

    print(f"Total files fixed: {total_fixed}/{len(txt_files)}")


if __name__ == '__main__':
    fix_labels(
        images_dir='data/dataset/aws/train/images',
        labels_dir='data/dataset/aws/train/labels',
        output_dir=None  # overwrite in place
    )
    fix_labels(
        images_dir='data/dataset/aws/test/images',
        labels_dir='data/dataset/aws/test/labels',
        output_dir=None  # overwrite in place
    )
    # fix_labels(
    #     images_dir='data/dataset/aws/valid/images',
    #     labels_dir='data/dataset/aws/valid/labels',
    #     output_dir=None  # overwrite in place
    # )