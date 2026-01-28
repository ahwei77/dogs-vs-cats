import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image

class DogCatDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        cat_count = 0
        dog_count = 0


        # 轉化數據集 狗1 貓0
        for filename in sorted(os.listdir(img_dir)):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(img_dir, filename)
                # 狗貓標籤判定
                if filename.startswith('dog'):
                    label = 1
                    dog_count += 1
                elif filename.startswith('cat'):
                    label = 0
                    cat_count += 1
                else:
                    continue
                self.images.append(filepath)
                self.labels.append(label)
        print(f"從 {img_dir}  載入 {cat_count} 張貓咪圖片 跟 {dog_count} 張狗狗圖片 .")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    train_transform = transforms.Compose([
        #transforms.Resize((img_size, img_size)),
        # 數據增強
        # 水平翻轉、隨機旋轉、亮度對比度飽和度調整、隨機裁剪、隨機擦除

        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), value='random'),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(data_dir, 'train','train')


    train_dataset = DogCatDataset(train_path, transform=train_transform)
    val_dataset = DogCatDataset(train_path, transform=val_transform)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(0.7 * dataset_size)


    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:split]
    val_indices = indices[split:]


    train_loader = DataLoader(Subset(train_dataset, train_indices),batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_dataset, val_indices),batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader