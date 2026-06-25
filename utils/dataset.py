from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class KermanyBinaryDataset(Dataset):
    """
    Binary OCT classifier dataset for Kermany/OCT2017.

    Label mapping:
        NORMAL -> 0 healthy
        CNV, DME, DRUSEN -> 1 unhealthy
    """

    def __init__(self, root_dir, split="train", image_size=224):
        self.root_dir = Path(root_dir)
        self.split = split

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.samples = []
        self._load_samples()

    def _find_split_dir(self):
        possible_dirs = [
            self.root_dir / "OCT2017" / self.split,
            self.root_dir / "oct2017" / self.split,
            self.root_dir / self.split,
            self.root_dir / "OCT2017 " / self.split,
        ]

        for d in possible_dirs:
            if d.exists():
                return d

        raise FileNotFoundError(
            f"Could not find split directory for split={self.split} under {self.root_dir}"
        )

    def _load_samples(self):
        split_dir = self._find_split_dir()

        class_dirs = {
            "NORMAL": 0,
            "CNV": 1,
            "DME": 1,
            "DRUSEN": 1,
        }

        for class_name, label in class_dirs.items():
            class_path = split_dir / class_name
            if not class_path.exists():
                print(f"[WARNING] Missing class folder: {class_path}")
                continue

            for img_path in class_path.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                    self.samples.append((img_path, label))

        print(f"[INFO] Loaded {len(self.samples)} samples from {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
