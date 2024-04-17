# import regex as re
import csv
import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch
import functools
import rdkit
import rdkit.Chem as Chem
import imblearn
# import h5py

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from matplotlib import colors

from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.Descriptors import MolWt

from PIL import Image
import PIL
import re

# check version number
# import imblearn
# from imblearn.over_sampling import RandomOverSampler
# oversample = RandomOverSampler(sampling_strategy='minority')


SMILES_COL_NAME = 'SMILES'

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
regex = re.compile(SMI_REGEX_PATTERN)


def tokenizer(smiles_string):
    tokens = [token for token in regex.findall(smiles_string)]
    return tokens


def atomwise_tokenizer(smi, exclusive_tokens=None):
    """Tokenizes a SMILES molecule at atom-level:
        (1) 'Br' and 'Cl' are two-character tokens
        (2) Symbols with bracket are considered as tokens
    exclusive_tokens: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'].
    Other symbols with bracket will be replaced by '[UNK]'. default is `None`."""

    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    if exclusive_tokens:
        for i, tok in enumerate(tokens):
            if tok.startswith('['):
                if tok not in exclusive_tokens:
                    tokens[i] = '[UNK]'
    return tokens


# 传进来多少创建多长的非重复vocab，数据不多则vocab也不全
def build_vocab(data):
    vocab_ = set()
    smiles = list(data[SMILES_COL_NAME])

    for ex in smiles:
        for letter in tokenizer(ex):
            vocab_.add(letter)

    vocab = {}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    for i, letter in enumerate(vocab_):
        vocab[letter] = i + 2
    inv_dict = {num: char for char, num in vocab.items()}
    inv_dict[0] = ''
    return vocab, inv_dict


# 与上面函数不同，直接手动创建一个完整的vocab
def custom_vocab():
    """Custom Vocab with certain commonly occuring molecules from DeepChem"""
    vocab_ = {'[c-]', '[SeH]', '[N]', '[C@@]', '[Te]', '[OH+]', 'n', '[AsH]', '[B]', 'b', '[S@@]', 'o', ')', '[NH+]',
              '[SH]', 'O', 'I', '[C@]', '-', '[As+]', '[Cl+2]', '[P+]', '[o+]', '[C]', '[C@H]', '[CH2]', '\\', 'P',
              '[O-]', '[NH-]', '[S@@+]', '[te]', '[s+]', 's', '[B-]', 'B', 'F', '=', '[te+]', '[H]', '[C@@H]', '[Na]',
              '[Si]', '[CH2-]', '[S@+]', 'C', '[se+]', '[cH-]', '6', 'N', '[IH2]', '[As]', '[Si@]', '[BH3-]', '[Se]',
              'Br', '[C+]', '[I+3]', '[b-]', '[P@+]', '[SH2]', '[I+2]', '%11', '[Ag-3]', '[O]', '9', 'c', '[N-]',
              '[BH-]', '4', '[N@+]', '[SiH]', '[Cl+3]', '#', '(', '[O+]', '[S-]', '[Br+2]', '[nH]', '[N+]', '[n-]', '3',
              '[Se+]', '[P@@]', '[Zn]', '2', '[NH2+]', '%10', '[SiH2]', '[nH+]', '[Si@@]', '[P@@+]', '/', '1', '[c+]',
              '[S@]', '[S+]', '[SH+]', '[B@@-]', '8', '[B@-]', '[C-]', '7', '[P@]', '[se]', 'S', '[n+]', '[PH]', '[I+]',
              '5', 'p', '[BH2-]', '[N@@+]', '[CH]', 'Cl'}
    vocab = {}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    for i, letter in enumerate(vocab_):
        vocab[letter] = i + 1
    inv_dict = {num: char for char, num in vocab.items()}
    inv_dict[0] = ''
    return vocab, inv_dict


def make_one_hot(data, vocab, max_len=120):  # 此处定义smiles最大长度
    """Converts the Strings to onehot data"""
    data_one_hot = np.zeros((len(data), max_len, len(vocab)))
    for i, smiles in enumerate(data):

        smiles = tokenizer(smiles)
        smiles = smiles[:120] + ['<PAD>'] * (max_len - len(smiles))

        for j, letter in enumerate(smiles):
            if letter in vocab.keys():
                data_one_hot[i, j, vocab[letter]] = 1
            else:
                data_one_hot[i, j, vocab['<UNK>']] = 1
    return data_one_hot


def oversample(input, labels):
    """Oversamples the input if there is a imbalanced data for QSAR"""
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_oversampled, y_oversampled = oversample.fit_resample(input, labels)
    return X_oversampled, y_oversampled


def get_ratio_classes(labels):
    """Returns the ratio of the labels in classification QSAR"""
    print('Number of 1s in dataset -- {} Percentage -- {:.3f}%'.format(labels[labels == 1].shape[0],
                                                                       labels[labels == 1].shape[0] / len(labels)))

    print('Number of 0s in dataset -- {} Percentage -- {:.3f}%'.format(labels[labels == 0].shape[0],
                                                                       labels[labels == 0].shape[0] / len(labels)))


def split_data(input, output, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(input, output,
                                                        test_size=test_size,
                                                        stratify=output,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def get_image(mol, atomset, name):
    """Save image of the SMILES for vis purposes"""
    hcolor = colors.to_rgb('green')
    if atomset is not None:
        # highlight the atoms set while drawing the whole molecule.
        img = MolToImage(mol, size=(600, 600), fitImage=True, highlightAtoms=atomset, highlightColor=hcolor)
    else:
        img = MolToImage(mol, size=(400, 400), fitImage=True)

    img = img.save(name + ".jpg")
    return img


def onehot_to_smiles(onehot, inv_vocab):
    """Converts Onehot output to smiles"""
    # return "".join(inv_vocab[let.item()] for let in onehot.argmax(axis=2)[0])
    return "".join(inv_vocab[let.item()] for let in onehot.argmax(dim=1))

# 返回 <UNK> 字符，表示一个未知或未出现的字符
def onehot_to_smiles_2(onehot, inv_vocab):
    """Converts Onehot output to smiles"""
    smiles = ""
    for let in onehot.argmax(dim=1):
        index = let.item()
        # 检查索引是否在 inv_vocab 范围内
        if index in inv_vocab:
            smiles += inv_vocab[index]
        else:
            smiles += "<UNK>"  # 或者选择其他的处理方式
    return smiles


def get_mol(smiles):
    """Returns SMILES String in RDKit molecule format"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def add_img(onehot, inv_vocab, name):
    smiles = onehot_to_smiles(onehot, inv_vocab)
    mol = get_mol(smiles)
    get_image(mol, {}, name)


def load_dataset(filename, split=True):
    # h5f = h5py.File(filename, 'r')
    with h5py.File(filename, 'r') as h5f:
        h5f = pd.read_hdf(filename, 'table')
    print(h5f)
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset = h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)


# 将loss保存在csv文件中
def save_loss_to_csv(loss_list, file_path):
    try:
        # loss = pd.read_csv(path
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['Loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for epoch, loss in enumerate(loss_list):
                writer.writerow({'Loss': loss})
    except Exception as e:
        print(f"An error occurred: {e}")
    # return


# 将输出张量转化成one-hot张量(torch张量)
def convert_to_binary_tensor(input_tensor):
    # 找到每行中最大值的索引
    max_indices = torch.argmax(input_tensor, dim=1, keepdim=True)
    # 创建一个全为0的二进制张量
    binary_tensor = torch.zeros_like(input_tensor)
    # 将每行中的最大值位置设为1
    binary_tensor.scatter_(1, max_indices, 1)
    # 确保数据类型为整数
    binary_tensor = binary_tensor.float()
    return binary_tensor


# 将输入和输出的SMILES保存在csv文件中
def save_smiles_to_csv(smiles_input, predictions_output, file_path):
    # 检查文件夹路径是否存在
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    df = pd.DataFrame({"Targets": smiles_input, "Predictions": predictions_output})
    df.to_csv(file_path, index=False)
    return


# 将输出张量换成类别张量进行输出比较
def tensor_to_list(tensor):
    row_indices = torch.argmax(tensor, dim=1)
    return row_indices.tolist()


# 将csv中的loss画图
def loss_figure():

    return


# 将csv中的index画图
def plot_index_prediction(file_dir: str) -> None:
    # 读取CSV文件
    data = pd.read_csv(file_dir)

    # 创建点图
    plt.figure(figsize=(10, 8))
    plt.scatter(data['Predictions'], data['Targets'], color='black')
    plt.title('Index_Prediction')
    plt.xlabel('Predictions')
    plt.ylabel('Targets')

    # 添加对角线
    min_value = min(data['Predictions'].min(), data['Targets'].min())
    max_value = max(data['Predictions'].max(), data['Targets'].max())
    plt.plot([min_value, max_value], [min_value, max_value], color='red')

    # 显示图表
    plt.show()
    plt.close()


# 将list中的index画图
def plot_index_list_prediction(inputs_index: list, outputs_index: list, save_dir: str = None) -> None:
    # 检查文件夹路径是否存在
    folder_path = os.path.dirname(save_dir)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # 寻找连续三个0的位置
    # start_index = 0
    # for i in range(len(inputs_index) - 1):
    #     if inputs_index[i] == 0 and inputs_index[i + 1] == 0 and inputs_index[i + 2] == 0:
    #         start_index = i + 1
    #         break
    # # 截断数据
    # inputs_index = inputs_index[:start_index]
    # outputs_index = outputs_index[:start_index]

    # 数据清洗：移除每个列表中含有0或1的元素对应的输入和输出
    cleaned_inputs = []
    cleaned_outputs = []

    # 遍历输入和输出列表的每个元素
    for input_item, output_item in zip(inputs_index, outputs_index):
        # 如果元素是子列表，进行数据清洗
        if isinstance(input_item, list) and isinstance(output_item, list):
            # 过滤掉子列表中的0和1
            cleaned_sublist_inputs = []
            cleaned_sublist_outputs = []
            for i, o in zip(input_item, output_item):
                if i not in [0, 1] and o not in [0, 1]:
                    cleaned_sublist_inputs.append(i)
                    cleaned_sublist_outputs.append(o)
            cleaned_inputs.extend(cleaned_sublist_inputs)
            cleaned_outputs.extend(cleaned_sublist_outputs)
        elif isinstance(input_item, int) and isinstance(output_item, int):
            # 如果元素是整数，且不是0或1，直接添加到绘图数据中
            if input_item not in [0, 1] and output_item not in [0, 1]:
                cleaned_inputs.append(input_item)
                cleaned_outputs.append(output_item)

    # 检查列表是否为空
    if not cleaned_inputs or not cleaned_outputs:
        print("输入或输出列表为空，无法绘图。")
        return

    # 创建点图
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(cleaned_outputs, cleaned_inputs, color='black')
    plt.title('Index_Prediction')
    plt.xlabel('Predictions')
    plt.ylabel('Targets')

    # 设置轴刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=18)

    # 添加对角线
    min_value = min(min(cleaned_outputs), min(cleaned_inputs))
    max_value = max(max(cleaned_outputs), max(cleaned_inputs))
    plt.plot([min_value, max_value], [min_value, max_value], color='red')

    # # 计算80%数据点覆盖区域
    # lower_bound = np.percentile(cleaned_outputs, 10)
    # upper_bound = np.percentile(cleaned_outputs, 90)
    # plt.axhline(y=lower_bound, color='blue', linestyle='--')
    # plt.axhline(y=upper_bound, color='blue', linestyle='--')
    # 计算最佳拟合线
    poly_coeff = np.polyfit(cleaned_outputs, cleaned_inputs, 1)
    poly_line = np.poly1d(poly_coeff)
    print(f"拟合线:", poly_line)
    # 绘制最佳拟合线
    # fit_line_x = np.array([min(cleaned_outputs), max(cleaned_outputs)])
    # fit_line_y = poly_line(fit_line_x)
    fit_line_x = np.linspace(min(cleaned_outputs), max(cleaned_outputs), 100)
    fit_line_y = poly_line(fit_line_x)
    # plt.plot(fit_line_x, fit_line_y, color='red', linewidth=2)

    # 计算置信区间线
    # 这里简化处理，仅用于演示，实际应用可能需要更严谨的统计方法
    # margin_of_error = np.std(cleaned_inputs - poly_line(cleaned_outputs)) * 1.96
    # plt.fill_between(fit_line_x, fit_line_y - margin_of_error, fit_line_y + margin_of_error, color='blue', alpha=0.1)
    # 填充两条线之间的区域
    # plt.fill_betweenx([min(cleaned_inputs), max(cleaned_inputs)], lower_bound, upper_bound, color='blue', alpha=0.2)

    # 获取最佳拟合线右上角的顶点坐标
    right_end_x = max(cleaned_outputs)
    right_end_y = poly_line(right_end_x)
    # 计算置信区间线的斜率
    angle_slope = poly_coeff[0]

    # 基于最佳拟合线右上角顶点的坐标计算置信区间
    error_margin = right_end_y * 0.1  # 误差范围设置为y值的10%

    # 计算并绘制置信区间线
    # 误差范围通过最佳拟合线的斜率和右上角顶点坐标来确定
    top_line_y = (fit_line_x - right_end_x) * angle_slope + (right_end_y + error_margin)
    bottom_line_y = (fit_line_x - right_end_x) * angle_slope + (right_end_y - error_margin)

    # 绘制置信区间的边界线
    plt.plot(fit_line_x, top_line_y, color='blue', linestyle='--')
    plt.plot(fit_line_x, bottom_line_y, color='blue', linestyle='--')

    # 填充置信区间线之间的区域
    plt.fill_between(fit_line_x, bottom_line_y, top_line_y, color='blue', alpha=0.1)
    # 统计不在拟合带内的点
    outside_band = np.logical_or(cleaned_inputs > poly_line(cleaned_outputs) + error_margin,
                                 cleaned_inputs < poly_line(cleaned_outputs) - error_margin)
    percentage_outside = np.mean(outside_band) * 100
    print(f"不在拟合带内的点占比: {percentage_outside:.2f}%")
    '''
    # 获取右上角的顶点坐标
    right_end_x = max(cleaned_outputs)
    right_end_y = poly_line(right_end_x)

    # 计算夹角线
    # 这里可以自行调整angle_offset来改变夹角线的斜率
    angle_offset = 0.1 * (max(cleaned_inputs) - min(cleaned_inputs))
    left_line_y = fit_line_y - angle_offset
    right_line_y = fit_line_y + angle_offset

    # 从最佳拟合线的右上角顶点向左下方扩展夹角线
    plt.fill_between(fit_line_x, left_line_y, right_line_y, color='blue', alpha=0.1)

    # 绘制夹角线的边界
    plt.plot(fit_line_x, left_line_y, color='blue', linestyle='--')
    plt.plot(fit_line_x, right_line_y, color='blue', linestyle='--')
    '''

    # 显示图表
    plt.show()
    print("成功绘图。")

    # 如果指定了保存路径，则保存图表
    if save_dir:
        plt.savefig(save_dir)

    plt.close()
# 示例代码，展示如何使用这个函数
# 假设inputs_index和outputs_index是两个列表，比如：
# inputs_index = [1, 2, 3, 4, 5]
# outputs_index = [1, 2, 1, 4, 5]
# 如果您想保存图表，可以提供一个保存路径，比如 '/path/to/save/plot.png'
# plot_index_prediction(inputs_index, outputs_index, '/path/to/save/plot.png')

def pac_figure():
    # 使用t-SNE进行降维
    latent_vectors_2d = TSNE(n_components=2).fit_transform(z.cpu().detach().numpy())
    # 绘制散点图
    plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1])
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('2D Visualization of the Latent Space')
    plt.show()
    plt.close()
    return


# 使用PCA或者t-SNE降维
def visualize_latent_space(latent_variables, use_tsne=False, perplexity=30, n_iter=300, save_path=None):
    # 通过use_tsne参数调整可视化方式
    if use_tsne:
        # t-SNE降维
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
        latent_transformed = tsne.fit_transform(latent_variables)
        title = 't-SNE on Latent Variables'
    else:
        # PCA降维
        pca = PCA(n_components=2)
        latent_transformed = pca.fit_transform(latent_variables)
        title = 'PCA on Latent Variables'

    # 绘制结果
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_transformed[:, 0], latent_transformed[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    if save_path is not None:
        plt.savefig(save_path)  # 保存图像
        plt.close()
    else:
        plt.show()  # 显示图像
        plt.close()


# 将所有epoch的数据整合再输入进上面的visualize_latent_space可视化函数
def collect_latent_variables(data_loader, model):
    model.eval()  # 确保模型处于评估模式
    latent_mu_list = []
    with torch.no_grad():  # 在此过程中不需要计算梯度
        for batch in data_loader:
            # 这里假设您的模型有一个名为'encode'的方法，该方法返回潜在空间的均值和日志方差
            mu, _ = model.encode(batch)
            latent_mu_list.append(mu.cpu().numpy())  # 将结果转换为numpy数组并存储
    # 将所有批次的潜在均值拼接在一起
    return np.concatenate(latent_mu_list, axis=0)


if __name__ == '__main__':
    data = pd.read_csv('./data/smiles_chembl.csv')
    data = data.tail(50000)
    print(data.head(10))

    vocab, inv_dict = build_vocab(data)
    print("Vocab", vocab)
    print("Vocab Size", len(vocab))

    vocab_2, inv_dict_2 = custom_vocab()
    print("Len of Custom Vocab", len(vocab_2))
    data_one_hot = make_one_hot(data[SMILES_COL_NAME], vocab)
    print(data_one_hot.shape)
    ####Checking onehot_to_smiles
    print("Original", data[SMILES_COL_NAME][5])
    print("Reconstructed", onehot_to_smiles(data_one_hot[5], inv_dict))
    print(data[SMILES_COL_NAME][5] == onehot_to_smiles(data_one_hot[5], inv_dict))
    #####

    add_img(data_one_hot[5], inv_dict, 'checking')

    print("One Hot Train Data Shape", data_one_hot.shape)
    print(data_one_hot[0])
