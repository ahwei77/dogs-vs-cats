import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
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
        for filename in os.listdir(img_dir):
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
        transforms.Resize((img_size, img_size)),

        # 數據增強
        # 水平翻轉、隨機旋轉、亮度對比度飽和度調整
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(data_dir, 'train','train')
    full_dataset = DogCatDataset(train_path, transform=None)
    
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_data, val_data = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) 
    )
    
    train_data.dataset.transform = train_transform 
    val_data.dataset.transform = val_transform     

    # Linux 環境下 num_workers 可設高一些提升效能
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader