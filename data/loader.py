from torch.utils.data import Dataset,DataLoader
import torch
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
import os
from sklearn.model_selection import train_test_split
from config import *

configs = configs()
dataset_name =configs.datasets
data_path = configs.data_path
batch_size = configs.batch_size
class CustomDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform
        if dataset_name== "normal":
            self.label_map = {'A01': 0, 'A02': 1, 'A03': 2, 'A04': 3, 'A05': 4, 'A06': 5, 'A07': 6, 'A08': 7, 'A09': 8,
                          'A10': 9, 'A11': 10, 'A12': 11}
        if dataset_name == "extreme":
            self.label_map = {'B01': 0, 'B02': 1, 'B03': 2, 'B04': 3, 'B05': 4, 'B06': 5, 'B07': 6}
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.image_list[idx]
        label_str = self.label_list[idx]
        label = self.label_map[label_str]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
def load_custom_data(root_dir, dataset_name = "normal",train_size=0.8, val_size=0.2,random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_list = []
    label_list = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if os.path.isfile(image_path) and not image_name.startswith('.'):
                    image_list.append(image_path)
                    label_list.append(class_name)
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_list, label_list, test_size=val_size , random_state=random_seed)
    trainset = CustomDataset(train_images, train_labels, transform=transform)
    valset = CustomDataset(val_images, val_labels, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader
