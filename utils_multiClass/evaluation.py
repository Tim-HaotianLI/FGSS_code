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


# ✅ 新增：多任务评估函数
def evaluate_multitask(y_true_1, y_pred_1, y_true_2, y_pred_2, num_classes_1=3, num_classes_2=3):
    """
    分别计算 Task1 / Task2 / 联合任务 的 TP,FP,FN,Precision,Recall,F1
    """

    # --- Task 1 ---
    p1, r1, f1_1 = calculate_metrics(y_true_1, y_pred_1, average='macro')
    TP1, FP1, FN1 = confusion_stats(y_true_1, y_pred_1, num_classes_1)

    # --- Task 2 ---
    p2, r2, f1_2 = calculate_metrics(y_true_2, y_pred_2, average='macro')
    TP2, FP2, FN2 = confusion_stats(y_true_2, y_pred_2, num_classes_2)

    # --- 合并任务 ---
    # 将 (y1,y2) 组合为联合标签，如 (1,0)->3, (2,2)->8 等
    combined_true = np.array(y_true_1) * num_classes_2 + np.array(y_true_2)
    combined_pred = np.array(y_pred_1) * num_classes_2 + np.array(y_pred_2)
    combined_classes = num_classes_1 * num_classes_2

    p_c, r_c, f1_c = calculate_metrics(combined_true, combined_pred, average='macro')
    TPc, FPc, FNc = confusion_stats(combined_true, combined_pred, combined_classes)

    results = {
        "task1": {
            "precision": p1, "recall": r1, "f1": f1_1,
            "TP": TP1, "FP": FP1, "FN": FN1
        },
        "task2": {
            "precision": p2, "recall": r2, "f1": f1_2,
            "TP": TP2, "FP": FP2, "FN": FN2
        },
        "combined": {
            "precision": p_c, "recall": r_c, "f1": f1_c,
            "TP": TPc, "FP": FPc, "FN": FNc
        }
    }

    return results


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