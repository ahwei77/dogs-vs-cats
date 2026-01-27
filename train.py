import torch
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  
from src.dataset import get_dataloaders
from src.model import get_model
from src.utils import plot_learning_curves
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    # 環境設定與目錄建立
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_weight_dir = os.path.join(base_dir, 'model_weight')
    os.makedirs(model_weight_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE = 32

    # 載入資料與模型
    data_path = os.path.join(base_dir, 'data')
    train_loader, val_loader = get_dataloaders(data_path, BATCH_SIZE)
    model = get_model().to(device)
    
    # 定義優化器與損失函數
    # 加入 weight_decay 降地過擬合
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # 訓練參數
    num_epochs = 120
    best_val_loss = float('inf')
    save_path = os.path.join(model_weight_dir, 'best_model.pth')
    
    # 紀錄歷史數據
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    print(f"開始訓練於 {device}")

    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # tqdm進度條
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            # 計算訓練 Accuracy
            preds = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            # 即時顯示 Loss 與 Accuracy
            current_acc = 100 * train_correct / train_total
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100 * train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        plot_learning_curves(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
                outputs = model(imgs)
                v_loss = criterion(outputs, labels)
                val_loss += v_loss.item()
                
                # 計算驗證Accuracy
                preds = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_pbar.set_postfix(v_loss=f"{v_loss.item():.4f}")
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        # 每十輪儲存權重
        if (epoch + 1) % 10 == 0:
            epoch_save_path = os.path.join(model_weight_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), epoch_save_path)
        # 儲存最佳模型
        print(f"\nSummary - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f">>> 發現更佳模型，已儲存至 {save_path}\n")
        else:
            print(f">>> 表現未提升。\n")
    plot_learning_curves(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])

if __name__ == "__main__":
    main()