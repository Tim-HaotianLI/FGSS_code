from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_metrics(y_true, y_pred, average='macro'):
    """
    y_true, y_pred: list 或 1D tensor
    average: 'macro' | 'micro' | 'weighted' 等
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return precision, recall, f1


def confusion_stats(y_true, y_pred, num_classes):
    """
    返回每个类别的 TP, FP, FN
    """
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)

    for cls in range(num_classes):
        TP[cls] = ((y_true == cls) & (y_pred == cls)).sum()
        FP[cls] = ((y_true != cls) & (y_pred == cls)).sum()
        FN[cls] = ((y_true == cls) & (y_pred != cls)).sum()
    return TP, FP, FN


def plot_confusion_matrix(all_labels, all_preds, class_names):
    """
    all_labels, all_preds: list or numpy array
    class_names: 类别名列表，例如 ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    """
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.show()