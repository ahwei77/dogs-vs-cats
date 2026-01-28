# 🐱🐶 Cat vs. Dog Image Classification with ResNet18

本專案是一個基於 PyTorch 的二元影像分類系統，使用 **ResNet18** 作為骨幹網路。專案特別針對模型泛化能力進行優化，實作了「訓練/驗證資料增強隔離」策略，並內建自動化的錯誤分析工具，能有效解決過擬合問題並提升辨識準確度。

## 📂 專案結構 (Directory Structure)

請確保您的目錄結構如下，以確保程式順利執行：

```text
Project_Root/
├── data/
│   └── train/
│       └── train/          # 存放 25,000 張貓狗圖片 (cat.x.jpg, dog.x.jpg)
├── model_weight/           # (自動建立) 存放訓練好的模型權重
├── error_analysis/         # (自動建立) 存放評估時抓出的錯誤圖片
├── results.png             # (自動產出) 混淆矩陣與 ROC 曲線圖
├── learning_curves.png     # (自動產出) 訓練過程 Loss/Acc 曲線圖
├── train.py                # 訓練主程式
├── evaluate.py             # 評估與錯誤分析程式
├── requirements.txt        # 環境依賴套件清單
├── src/
│   ├── dataset.py          # 數據讀取、增強變換、索引切分
│   ├── model.py            # ResNet18 模型定義 (含 Dropout 優化)
│   └── utils.py            # 繪圖與指標計算工具
└── README.md