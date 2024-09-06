import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class LabeledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        label_map = {'Malignant': 0, 'Benign': 1, 'Normal': 2}
        for label_name, label_index in label_map.items():
            class_dir = os.path.join(self.root_dir, label_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(label_index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self._load_data()

    def _load_data(self):
        for img_name in os.listdir(self.root_dir):
            img_path = os.path.join(self.root_dir, img_name)
            self.images.append(img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(0)  # Returning a dummy label (e.g., tensor(0))
