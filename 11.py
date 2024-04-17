# -*- coding:utf-8 -*-
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

# 文件路径
file_path = r"D:\学习\研三\小论文第二篇（中文）\化工学报\补充数据结果\output\inputs_and_recon_last_epoch_2.npz"

# 加载数据
data = np.load(file_path)
inputs = data['inputs'] # 真实标签
input_recon = data['input_recon'] # 预测值

# 您感兴趣的类别
interested_classes = [28, 42, 50, 66, 92]

# 初始化空列表来存储每个类别的真实标签和预测概率
true_labels = {cls: [] for cls in interested_classes}
pred_probs = {cls: [] for cls in interested_classes}

# 处理数据
for i in range(inputs.shape[0]):  # 遍历每个batch
    for j in range(inputs.shape[1]):  # 遍历每个样本
        true_class = np.argmax(inputs[i, j, :])  # 真实类别
        for cls in interested_classes:
            true_labels[cls].append(int(true_class == cls))
            pred_probs[cls].append(input_recon[i, j, cls])

# 确保列表中的数据转换为numpy数组
for cls in interested_classes:
    true_labels[cls] = np.array(true_labels[cls])
    pred_probs[cls] = np.array(pred_probs[cls])

# 设置全局阈值
threshold = 0.000000000001  # 您可以根据需要调整这个值

# 应用阈值并转换预测概率为二元标签
binary_pred_probs = {cls: (pred_probs[cls] >= threshold).astype(int) for cls in interested_classes}

# 绘制ROC曲线
plt.figure(figsize=(10, 8))  # 设置图像大小

for cls in interested_classes:
    # 计算每个类别的ROC曲线和AUC
    fpr, tpr, _ = roc_curve(true_labels[cls], pred_probs[cls])
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, linewidth=2, label=f'Class {cls} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for Selected Classes')
# plt.legend(loc="lower right", frameon=False, fontsize=18)
# 将图例移动到左上角靠近对角线的位置
plt.legend(loc='upper left', bbox_to_anchor=(0.33, 0.33), frameon=False, fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)

# 获取当前的坐标轴, 并加粗横纵坐标轴线
ax = plt.gca()
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# 保存高分辨率图像
plt.savefig("roc_curves.png", dpi=600)

plt.show()

# 检查每个类别的正样本数量
for cls in interested_classes:
    print("Number of positive samples for class {}: {}".format(cls, sum(true_labels[cls])))

# 选择一个类别进行调试，确保该类别在 interested_classes 中
cls_debug = interested_classes[0]  # 这里选择列表中的第一个类别

# 打印该类别的一些真实标签和预测概率
print("Some true labels for class {}:".format(cls_debug), true_labels[cls_debug][:10])
print("Some predicted probabilities for class {}:".format(cls_debug), pred_probs[cls_debug][:10])

