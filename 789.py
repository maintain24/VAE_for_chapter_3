from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier  # 示例分类器
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from typing import List

"""
画ROC曲线
"""
def plot_multiclass_roc(y_true, y_pred, n_classes, classes):
    # 二值化真实标签
    y_true = label_binarize(y_true, classes=list(range(n_classes)))

    # 构造简化的预测概率
    y_score = np.zeros((y_pred.size, n_classes))
    for i in range(y_pred.size):
        y_score[i, y_pred[i]] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'cyan', 'magenta', 'orange', 'grey', 'brown'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC')
    plt.legend(loc="lower right")
    plt.show()


def plot_multiclass_roc_2(y_true, y_pred, selected_classes):
    # 二值化真实标签
    y_true_binary = label_binarize(y_true, classes=range(max(selected_classes) + 1))

    # 构造简化的预测概率
    n_classes = max(selected_classes) + 1
    y_score = np.zeros((len(y_pred), n_classes))
    for i in range(len(y_pred)):
        if y_pred[i] in selected_classes:
            y_score[i, y_pred[i]] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for class_label in selected_classes:
        fpr[class_label], tpr[class_label], _ = roc_curve(y_true_binary[:, class_label], y_score[:, class_label])
        roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])

    # 绘制ROC曲线
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'cyan', 'magenta', 'orange', 'grey', 'brown'])
    for class_label, color in zip(selected_classes, colors):
        plt.plot(fpr[class_label], tpr[class_label], color=color, lw=2,
                 label=f'ROC curve of class {class_label} (area = {roc_auc[class_label]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Selected Multi-Class ROC')
    plt.legend(loc="lower right")
    plt.show()



"""
预测类别进行画图，其实应该使用混淆矩阵
# 读取数据
output_file = r"D:\\学习\\研三\\小论文第二篇（中文）\\化工学报\\补充数据结果\\all.csv"
df = pd.read_csv(output_file)
y_true = df['Targets']
y_pred = df['Predictions']

# 确定类别数量
max_class_index = max(y_true.max(), y_pred.max())
n_classes = max_class_index + 1
classes = [f'Class {i}' for i in range(n_classes)]

# 检查 y_pred 中的值是否有效
if y_pred.max() >= n_classes or y_pred.min() < 0:
    raise ValueError("y_pred contains invalid class indices.")

# 指定感兴趣的类别
interested_classes = [30, 31, 32]  # 例如，只关注类别 0、1、2

# 根据感兴趣的类别过滤数据
filtered_indices = [i for i, cls in enumerate(classes) if cls in interested_classes]
filtered_classes = [classes[i] for i in filtered_indices]

# 根据感兴趣的类别构建新的 y_true 和 y_score
filtered_y_true = y_true[:, filtered_indices]
filtered_y_score = y_pred[:, filtered_indices]

# 重新绘制 ROC 曲线
plot_multiclass_roc(filtered_y_true, filtered_y_score, len(interested_classes), filtered_classes)

# 绘制 ROC 曲线
# plot_multiclass_roc(y_true, y_pred, n_classes, classes)
"""

# 使用混淆矩阵画图
# 假设 last_epoch_inputs 和 last_epoch_input_recon 是我们之前保存的数据
data = np.load(r'D:\学习\研三\小论文第二篇（中文）\化工学报\补充数据结果\output\inputs_and_recon_last_epoch.npz')
inputs = torch.tensor(data['inputs'])  # 真实标签的混淆矩阵
outputs = torch.tensor(data['input_recon'])  # 预测标签的混淆矩阵

# 提取预测标签
y_pred = torch.argmax(outputs, dim=2).reshape(-1)  # 展平为一维数组

# 提取真实标签
# 在此假设每个混淆矩阵的对角线代表正确的分类情况
y_true = torch.argmax(inputs, dim=2).reshape(-1)  # 展平为一维数组

# 二值化真实标签
n_classes = 137  # 类别数量
y_true_binary = label_binarize(y_true, classes=range(n_classes))

# 指定感兴趣的类别
interested_classes = [20, 21, 22]
# 将感兴趣的类别转换为张量
interested_classes_tensor = torch.tensor(interested_classes)

# 过滤出仅包含感兴趣类别的样本
indices_of_interest = (y_true.unsqueeze(1) == interested_classes_tensor).any(1)
filtered_y_true = y_true[indices_of_interest]
filtered_y_pred = y_pred[indices_of_interest]

# 对过滤后的真实标签进行二值化
filtered_y_true_binary = label_binarize(filtered_y_true, classes=interested_classes)

# 继续执行绘制ROC曲线的函数
plot_multiclass_roc_2(filtered_y_true_binary, filtered_y_pred, [2])

'''
# 示例数据
y_true = [0, 2, 1, 2, 2]  # 真实标签
y_score = np.array([[0.5, 0.2, 0.3], [0.1, 0.3, 0.6], [0.2, 0.4, 0.4], [0.1, 0.2, 0.7], [0.2, 0.2, 0.6]])  # 预测概率
n_classes = 3
classes = ['Class 0', 'Class 1', 'Class 2']

# 绘制ROC曲线
plot_multiclass_roc(y_true, y_score, n_classes, classes)
'''
