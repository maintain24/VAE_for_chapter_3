import pandas as pd
import os


def clean_and_merge_csv(folder_path: str, output_file: str) -> None:
    all_targets = []
    all_predictions = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            # 读取 CSV 文件
            data = pd.read_csv(file_path)

            # 遍历每一行，清洗数据并追加到列表中
            for _, row in data.iterrows():
                target, prediction = row["Targets"], row["Predictions"]
                if target not in [0, 1] and prediction not in [0, 1]:
                    all_targets.append(target)
                    all_predictions.append(prediction)

    # 将结果保存到新的 CSV 文件
    df = pd.DataFrame({'Targets': all_targets, 'Predictions': all_predictions})
    df.to_csv(os.path.join(folder_path, output_file), index=False)


folder_path = r"D:\\学习\\研三\\小论文第二篇（中文）\\化工学报\\补充数据结果\\smiles_index"
output_file = r"D:\\学习\\研三\\小论文第二篇（中文）\\化工学报\\补充数据结果\\all.csv"
clean_and_merge_csv(folder_path, output_file)
