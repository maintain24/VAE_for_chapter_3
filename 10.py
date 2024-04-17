# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

"""
对路径上的csv文件进行指数式平滑
"""

# 设置文件路径
file_path = r"D:\学习\研三\小论文第二篇（中文）\化工学报\补充数据结果\step_loss\step_loss_CNC.csv"

# 读取CSV文件
try:
    data = pd.read_csv(file_path)

    # 应用指数平滑
    # alpha是平滑因子，这里取alpha=2/(span+1)，span相当于窗口大小
    span = 10
    data['Smooth_Loss'] = data.iloc[:, 2].ewm(span=span, adjust=False).mean()

    # 绘制原始数据和平滑后数据的折线图
    plt.figure(figsize=(10, 5))
    plt.plot(data.iloc[:, 0], label='Original Loss', color='blue', linestyle='-', alpha=0.6)
    plt.plot(data['Smoothed_Loss'], label='Smoothed Loss', color='red', linestyle='--')
    plt.legend()
    plt.title('Loss vs. Smoothed Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    # 保存修改后的DataFrame到新的CSV文件
    new_file_path = file_path.replace('loss.csv', 'smoothed_loss.csv')
    data.to_csv(new_file_path, index=False)
except Exception as e:
    print(f"An error occurred: {e}")