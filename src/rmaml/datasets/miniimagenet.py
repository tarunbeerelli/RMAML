"""
MiniImageNet dataset loader — pickle format.

Expected files:
    data/miniimagenet/
        mini-imagenet-cache-train.pkl
        mini-imagenet-cache-val.pkl
        mini-imagenet-cache-test.pkl

Each pickle contains a dict:
    {
        "image_data": np.array of shape (N, 84, 84, 3),
        "class_dict": {class_name: [list of indices]}
    }
"""

import pickle
from pathlib import Path

import torch
from torchvision import transforms


def build_transform(augment: bool = False) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if augment:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(84),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def load_miniimagenet(
    root: str,
    split: str = "train",
    augment: bool = False,
) -> dict[int, list[torch.Tensor]]:
    """
    Load MiniImageNet pickle split into a dataset_map.

    Returns:
        dict mapping class_id (int) → list of image tensors (3, 84, 84)
    """
    root = Path(root)
    pkl_path = root / f"mini-imagenet-cache-{split}.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    image_data = data["image_data"]  # (N, 84, 84, 3) uint8
    class_dict = data["class_dict"]  # {class_name: [indices]}

    transform = build_transform(augment=augment)

    dataset_map: dict[int, list[torch.Tensor]] = {}
    for class_id, (class_name, indices) in enumerate(class_dict.items()):
        dataset_map[class_id] = [transform(image_data[i]) for i in indices]

    n_classes = len(dataset_map)
    n_images = sum(len(v) for v in dataset_map.values())
    print(f"Loaded MiniImageNet {split}: {n_classes} classes, {n_images} images")

    return dataset_map
