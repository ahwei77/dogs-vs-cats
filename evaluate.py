import shutil
import torch
import os
from src.dataset import get_dataloaders
from src.model import get_model
from src.utils import plot_metrics_and_roccurve

def main():

    # 僅計算指標 0 輸出錯誤圖片 1
    EXPORT_ERRORS = 0

    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(base_dir, 'data')
    model_path = os.path.join(base_dir, 'model_weight', 'best_model.pth')
    
    _, val_loader = get_dataloaders(data_path)

    model = get_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    if EXPORT_ERRORS:
        error_dir = os.path.join(base_dir, 'error_analysis')
        if os.path.exists(error_dir):
            shutil.rmtree(error_dir)
        os.makedirs(os.path.join(error_dir, 'cat_as_dog'), exist_ok=True)
        os.makedirs(os.path.join(error_dir, 'dog_as_cat'), exist_ok=True)
        
        # 為了溯源圖片路徑，需要存取底層 dataset
        val_indices = val_loader.dataset.indices
        original_dataset = val_loader.dataset.dataset

    y_true, y_probs = [], []
    print(f"正在評估驗證集圖片 ( {'抓取錯誤圖片' if EXPORT_ERRORS else '指標計算'})...")

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = outputs.cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            actuals = labels.numpy().astype(int)
            
            y_probs.extend(probs)
            y_true.extend(actuals)
            
            # 處理錯誤圖片輸出
            if EXPORT_ERRORS:
                for j in range(len(actuals)):
                    if preds[j] != actuals[j]:
                        # 計算該圖在全部圖片中的索引
                        global_idx = val_indices[i * val_loader.batch_size + j]
                        img_path = original_dataset.images[global_idx]
                        img_name = os.path.basename(img_path)
                        
                        # 判定分類並複製檔案
                        sub_folder = 'cat_as_dog' if actuals[j] == 0 else 'dog_as_cat'
                        target_path = os.path.join(error_dir, sub_folder, img_name)
                        shutil.copy(img_path, target_path)

    print("\n評估完成")
    if EXPORT_ERRORS:
        print(f"辨識錯誤的圖片已儲存至: {os.path.join(base_dir, 'error_analysis')}")
    
    plot_metrics_and_roccurve(y_true, y_probs)

if __name__ == "__main__":
    main()