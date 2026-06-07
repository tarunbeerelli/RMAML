"""
MiniImageNet dataset loader.

Expected folder structure after download:
    data/miniimagenet/
        images/          ← all 60,000 images as .jpg
        train.csv        ← filename, label columns
        val.csv
        test.csv

Download instructions:
    1. kaggle datasets download -d arjunashok33/miniimagenet
    2. unzip into data/miniimagenet/

Split: 64 train / 16 val / 20 test classes (Ravi & Larochelle 2017)
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def build_transform(augment: bool = False, image_size: int = 84) -> transforms.Compose:
    """
    Standard MiniImageNet transforms.
    augment=True  → random crop + flip (meta-train only)
    augment=False → centre crop only (meta-val, meta-test)
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def load_miniimagenet(
    root: str,
    split: str = "train",  # "train" | "val" | "test"
    augment: bool = False,
) -> dict[int, list[torch.Tensor]]:
    """
    Load MiniImageNet split into a dataset_map.

    Returns:
        dict mapping class_id (int) → list of image tensors (3, 84, 84)
    """
    root = Path(root)
    csv_path = root / f"{split}.csv"
    images_dir = root / "images"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Download MiniImageNet and place CSVs in {root}"
        )

    transform = build_transform(augment=augment)

    # Build label → int mapping from CSV
    label_to_id: dict[str, int] = {}
    rows: list[tuple[str, str]] = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            label = row["label"]
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)
            rows.append((fname, label))

    # Load images into dataset_map
    dataset_map: dict[int, list[torch.Tensor]] = {
        i: [] for i in range(len(label_to_id))
    }

    for fname, label in rows:
        img_path = images_dir / fname
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img)
        dataset_map[label_to_id[label]].append(tensor)

    n_classes = len(dataset_map)
    n_images = sum(len(v) for v in dataset_map.values())
    print(f"Loaded MiniImageNet {split}: {n_classes} classes, {n_images} images")

    return dataset_map
