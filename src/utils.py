import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import os

based_dir= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def plot_metrics_and_roccurve(y_true, y_probs):
    y_true, y_probs = np.array(y_true), np.array(y_probs)
    y_pred = (y_probs > 0.5).astype(int)

    print("\n評估結果:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # 混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    ax1.set_title('Confusion Matrix')
    # ROC曲線
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    ax2.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    ax2.set_title('ROC Curve')
    ax2.legend()
    save_path = os.path.join(based_dir, 'results.png') 
    plt.savefig(save_path)
    print(f"評估圖表已儲存至: {save_path}")


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    # 畫出 Loss 曲線
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()

    # 畫出 Accuracy 曲線
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(based_dir, 'learning_curves.png')
    plt.savefig(save_path)
    print(f"學習曲線已儲存至: {save_path}")
