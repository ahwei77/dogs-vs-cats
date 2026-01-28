import torch.nn as nn
from torchvision import models

def get_model():
    # 不使用resnet50遷移學習，改成重新訓練參數 改用resnet18比較快
    model = models.resnet18(weights=None)
    # 修改最後的全連接層二元分類任務
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )

    return model