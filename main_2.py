from model_2 import *
from utils import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import torch
import torch.optim as optim
import random
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.autograd import gradcheck
import torch.backends.cudnn as cudnn
from args import get_arguments
from dataset import VAE_Dataset

# torch.backends.cudnn.enabled = False

NUM_EPOCHS = 30
BATCH_SIZE = 64  # 原本是256
LATENT_DIM = 592  # 原本是292
RANDOM_SEED = 42
LR = 0.0001
DYN_LR = True
EVAL_PATH = './Save_Models/189checkpoint.pth'
# 获取当前日期
current_date = datetime.now().strftime("%Y%m%d")

args = get_arguments()
device = args.device

# 打印命令行参数（即运行脚本的命令）
print("运行命令:", ' '.join(sys.argv))


def main():
    # data = pd.read_csv(args.data, nrows=2000)  # 取前2000-4000行，总共50000行
    # data = pd.read_csv(args.data, skiprows=range(1, 1999), nrows=2000)  # 取前2000-4000行，总共50000行
    data = pd.read_csv(args.data, skiprows=range(2000, 3999), nrows=2000)  # 取前2000-4000行，总共50000行

    smiles = data[SMILES_COL_NAME]
    # labels = np.array(data['p_np'])
    # labels = np.zeros((len(smiles), 1))  # 勾史代码
    labels = data['Performances']
    print("Example Smiles", smiles[0:10])

    ##Building the Vocab from DeepChem's Regex
    vocab, inv_dict = build_vocab(data)
    vocab_size = len(vocab)
    print('vocab_len:', vocab_size)  # len(vocab)=101
    print('vocab:', vocab)
    print(vocab.items())
    ##Converting to One Hot Vectors
    data = make_one_hot(data[SMILES_COL_NAME], vocab)
    print("Input Data Shape", data.shape)

    data_val = pd.read_csv(args.val_data, nrows=512)  # 取前50行，原本是data_val = pd.read_csv(args.val_data)
    smiles_val = data_val[SMILES_COL_NAME]
    # labels_val = np.zeros((len(smiles), 1))  # 勾史代码
    labels_val = data_val['Performances']
    data_val = make_one_hot(data_val[SMILES_COL_NAME], vocab)

    X_train = data
    X_test = data_val
    y_train = labels
    y_test = labels_val

    # 创建自定义Dataset实例
    train_dataset = VAE_Dataset(data=X_train, labels=y_train)
    val_dataset = VAE_Dataset(data=X_test, labels=y_test)

    ##Checking ratio for Classification Datasets
    # print("Ratio of Classes")
    # get_ratio_classes(labels)

    ##To oversample datasets if dataset is imbalanced change to True
    Oversample = False
    if Oversample:
        print(data.shape, labels.shape)
        data, labels = oversample(data, labels)
        print("After Over Sampling")
        # get_ratio_classes(labels_oversampled)
        get_ratio_classes(labels)

    ##Train Test Split

    # X_train, X_test, y_train, y_test = split_data(data,labels)
    # print("Train Data Shape--{} Labels Shape--{} ".format(X_train.shape,y_train.shape))
    # print("Test Data Shape--{} Labels Shape--{} ".format(X_test.shape,y_test.shape))

    use_vae = args.model_type == 'mol_vae'
    if use_vae:
        ##Using Molecular VAE Arch as in the Paper with Conv Encoder and GRU Decoder
        # input_channels = X_train.shape[-1] + 1  # 对应Conv_Encoder
        # enc = Conv_Encoder(vocab_size, input_channels=input_channels).to(device)
        input_channels = X_train.shape[-1] + 1
        enc = GRU_Encoder(vocab_size, input_channels=input_channels).to(device)
        dec = GRU_Decoder(vocab_size, args.latent_dim, num_layers=3).to(device)  # 增加了, num_layers=3
        # dec = GRU_Decoder_2(vocab_size, args.latent_dim, gru_hidden_dim=128, num_layers=3).to(device)  # 增加了, num_layers=3
        model = Molecule_VAE(enc, dec, device, args.latent_dim).to(device)
        model.get_num_params()
        # 设置模型参数的 requires_grad 为 True (不过似乎没啥用)
        for param in model.parameters():
            param.requires_grad = True

        # 打印模型的权重和偏置参数维度
        for name, param in model.state_dict().items():
            print(f"Parameter: {name}, Size: {param.size()}")
    else:
        # Using FC layers for both Encoder and Decoder
        input_dim = 120 * 71
        hidden_dim = 200
        hidden_2 = 120
        latent = 60
        enc = Encoder(input_dim, hidden_dim, hidden_2)
        dec = Decoder(input_dim, hidden_dim, latent)
        model = VAE(enc, dec, latent)
        model.get_num_params()


    # TODO: Add loading function
    # if os.path.isfile(args.model):
    #    model.load(charset, args.model, latent_rep_size = args.latent_dim)

    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if DYN_LR:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                         factor=0.8,
                                                         patience=3,
                                                         min_lr=0.0001)

    # dataloader = torch.utils.data.DataLoader(X_train,
    #                                          batch_size=args.batch_size,
    #                                          shuffle=True,
    #                                          num_workers=3,
    #                                          drop_last=True)  # num_workers=6
    # 下面的dataloader中用的dataset加入了性能标签数据label
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=3,
                                             drop_last=True)  # num_worker=6

    # val_dataloader = torch.utils.data.DataLoader(X_test,
    #                                              batch_size=args.batch_size,
    #                                              shuffle=True,
    #                                              num_workers=3,
    #                                              drop_last=True)  # num_workers=6
    # 下面的dataloader中用的dataset加入了性能标签数据label
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=3,
                                                 drop_last=True)  # num_worker=6

    best_epoch_loss_val = 100000
    x_train_data_per_epoch = X_train.shape[0] - X_train.shape[0] % args.batch_size
    x_val_data_per_epoch = X_test.shape[0] - X_test.shape[0] % args.batch_size
    print("Div Quantities", x_train_data_per_epoch, x_val_data_per_epoch)
    print()
    print("###################################################################################")
    # 新增：创建两个列表，用于存储每个step过程的loss值和每个epoch的loss
    step_losses = []
    all_losses = []
    inputs_smiles_list = []
    outputs_smiles_list = []

    # 初始化一个列表来收集采样的z值,用于PAC或者t-SNE可视化
    sampled_z_values = []
    # 设置采样频率，例如每处理10个批次采样一次，避免可视化数据太大内存不够
    sampling_frequency = 10

    # 初始化用于存储整个epoch数据的input和output变量
    all_inputs = []
    all_outputs = []
    last_epoch_inputs = None  # 用于存储最后一个epoch的inputs
    last_epoch_input_recon = None  # 用于存储最后一个epoch的input_recon

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_sampled_z_values = []  # 存储当前epoch的z值,用于PAC降维
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Epoch -- {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(epoch))

        for i, (features, labels) in tqdm(enumerate(dataloader)):
            inputs = features.float().to(device)
            targets = labels.float().to(device)
            # print('labels shape:', targets.shape)  # torch.Size([32])
            # print("Inputs Shape:", inputs.shape)  # torch.Size([512, 120, 101])
            # 调整 inputs 的形状，保持 batch 维度，将其它维度展平
            # inputs_flat = inputs.permute(0, 2, 1)  # torch.Size([512, 101, 120])
            # inputs_flat = inputs.view(-1, inputs.size(-1))
            inputs_flat = inputs
            # 原本是inputs = inputs.reshape(batch_size, -1).float()
            optimizer.zero_grad()

            input_recon, z = model(inputs_flat, targets)

            # 添加代码以收集预测和目标
            all_inputs.extend(tensor_to_list(inputs))
            all_outputs.extend(tensor_to_list(input_recon))
            """使用convert_to_binary_tensor函数，
            例如outputs_onehot = convert_to_binary_tensor(input_recon[i])
            最好是能把几个dataloader的也汇总再一个图里，不行也没关系，足够了"""

            # 根据设定的频率采样z值
            if i % sampling_frequency == 0:
                print(f"Before appending, 'epoch_sampled_z_values' is a {type(epoch_sampled_z_values)}")
                z_numpy = z.detach().cpu().numpy()
                print(f"Shape of z_numpy: {z_numpy.shape}")  # 打印z_numpy的形状
                # print(f'z_numpy tensor: ', z_numpy)  # 打印z_numpy
                epoch_sampled_z_values.append(z_numpy)

            # print("Outputs Shape:", input_recon.shape)
            # print('output:', input_recon)  # 打印输出张量
            latent_loss_val = latent_loss(model.z_mean, model.z_sigma)  # *100
            print('latent_loss_val:', latent_loss_val)
            # ①原本是
            # loss = F.binary_cross_entropy(input_recon, inputs, reduction='sum') + latent_loss_val
            # (input_recon, inputs, reduction='sum')等同于原本的(input_recon, inputs, size_average=False)  # 也可以是'mean'
            loss = F.binary_cross_entropy(input_recon, inputs, reduction='sum') + latent_loss_val
            # loss = F.binary_cross_entropy_with_logits(inputs_flat.reshape(-1, inputs_flat.size(-1)).float(),
            #                                           input_recon.reshape(-1, input_recon.size(-1)),) + latent_loss_val

            # criterion = nn.MSELoss()
            # criterion = nn.CrossEntropyLoss()
            # criterion = nn.NLLLoss()
            criterion = nn.CrossEntropyLoss()

            # 查看输入和输出张量经过argmax后的位置索引
            # print('shuru_loss', torch.argmax(inputs_flat, dim=2, keepdim=True))
            # print('shuchu_loss', torch.argmax(input_recon, dim=2, keepdim=True))

            # ②根据上面设置的criterion函数计算loss
            # loss = criterion(input_recon.float(), inputs_flat.float())

            # ③根据每行的最大值位置索引计算loss，但是argmax不可微分，无法计算loss.backward()
            # loss = F.cross_entropy(torch.argmax(input_recon, dim=2, keepdim=True).float(),
            #                        torch.argmax(inputs_flat, dim=2, keepdim=True).float(),
            #                        reduction='mean')
            #  F.binary_cross_entropy适用于二分类问题，尝试换cross_entropy
            #  使用argmax函数不可微分，导致loss.backward()无法计算梯度
            print('loss:', loss)
            print()

            # 记录每个step的loss
            step_losses.append(loss.detach().numpy())

            loss.backward()
            # 检查梯度
            # for name, param in model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         print(f'Gradient norm of {name}: {param.grad.norm().item()}')
            # gradcheck(model, inputs, eps=1e-6, atol=1e-4)  # 也是检查梯度的工具

            optimizer.step()
            epoch_loss += loss.item()

            # 保存input和output用来画ROC等指标
            if epoch == args.epochs - 1:
                # 只在最后一个epoch保存数据
                last_epoch_inputs = inputs.detach().cpu()
                last_epoch_input_recon = input_recon.detach().cpu()

        # 检查输出目录是否存在，如果不存在则创建
        output_dir = 'outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 训练结束后，将数据保存为CSV文件
        if last_epoch_inputs is not None and last_epoch_input_recon is not None:
            print("npz文件中Inputs维度:", last_epoch_inputs.shape)
            print("npz文件中output维度:", last_epoch_input_recon.shape)

            # 使用 numpy 的 savez 函数将数据保存为 .npz 文件
            np.savez(os.path.join(output_dir, 'inputs_and_recon_last_epoch.npz'), inputs=last_epoch_inputs.numpy(),
                     input_recon=last_epoch_input_recon.numpy())

        # 在最后一个epoch结束时进行可视化
        if epoch == args.epochs - 1:
            save_smiles_to_csv(all_inputs, all_outputs,
                               r'/mnt/pycharm_project_VAE/smiles_csv/{}/tensors_index.csv'.format(current_date))
            plot_index_list_prediction(all_inputs, all_outputs,
                                       r'/mnt/pycharm_project_VAE/index_fig/{}/All_Figure.jpg'.format(current_date))

        # 记录每个 epoch 的 loss
        all_losses.append((epoch_loss / x_train_data_per_epoch))
        # 保存 loss 列表到指定位置
        save_loss_to_csv(step_losses, r'/mnt/pycharm_project_VAE/step_loss.csv')
        save_loss_to_csv(all_losses, r'/mnt/pycharm_project_VAE/epoch_loss.csv')

        # # 检查每个epoch内z值的形状是否一致
        # if len(set([x.shape for x in sampled_z_values])) != 1:
        #     raise ValueError("Inconsistent shape of z values within an epoch.")
        # # 将采样的z值连接成一个数组，用于PAC可视化
        # sampled_z_values = np.concatenate(sampled_z_values, axis=0)
        # # 使用PCA进行可视化
        # visualize_latent_space(sampled_z_values, use_tsne=False, save_path='pca_visualization.png')
        # 将该epoch中的所有z值拼接成一个NumPy数组
        epoch_z = np.concatenate(epoch_sampled_z_values, axis=0)
        print(f"Shape of z_numpy: {epoch_z.shape}")
        sampled_z_values.append(epoch_z)  # 将这个数组存储在另一个列表中
        # print(f"Shape of z_numpy: {sampled_z_values.shape}")

        print("Train Loss -- {:.3f}".format(epoch_loss / x_train_data_per_epoch))
        ###Add 1 Image per Epoch for Visualisation
        # 随机取样查看样本生成情况
        # data_point_sampled = random.randint(0, args.batch_size - 1)
        # 固定查看前五个样本生成情况，若大于1则加上for循环
        data_point_sampled = 1

        print("INPUT", inputs[data_point_sampled])
        # print('Input shape:', inputs[data_point_sampled].shape)  # torch.Size([120, 48])
        print("OUTPUT", input_recon[data_point_sampled])  # 原本是input_recon[data_point_sampled].reshape(1, 120, len(vocab))
        # print('Output shape:', input_recon[data_point_sampled].shape)  # torch.Size([120, 48])

        for i in range(25):  # 取决于想打印前几列
            inputs_smiles = onehot_to_smiles_2(inputs[i].cpu().detach(), inv_dict)
            outputs_onehot = convert_to_binary_tensor(input_recon[i])
            outputs_smiles = onehot_to_smiles_2(outputs_onehot.cpu().detach(), inv_dict)
            inputs_smiles_list.append(inputs_smiles)
            outputs_smiles_list.append(outputs_smiles)
            print("Input smiles-- ", inputs_smiles)
            print("Output smiles-- ", outputs_smiles)
            save_smiles_to_csv(inputs_smiles_list, outputs_smiles_list, r'/mnt/pycharm_project_VAE/smiles.csv')
            # 记录索引，保存
            inputs_index = tensor_to_list(inputs[i])
            outputs_index = tensor_to_list(outputs_onehot)
            save_smiles_to_csv(inputs_index, outputs_index,
                               r'/mnt/pycharm_project_VAE/smiles_csv/{}/tensors_index_{}.csv'.format(current_date, i))
            plot_index_list_prediction(inputs_index, outputs_index,
                                       r'/mnt/pycharm_project_VAE/index_fig/{}/{}Figure.jpg'.format(current_date, i))

        '''
        # print('input dtype:', len(inputs[data_point_sampled]))  # 查看输入格式  .reshape(1, 120, len(vocab))
        inputs_smiles = onehot_to_smiles(inputs[data_point_sampled].cpu().detach(), inv_dict)
        outputs_onehot = convert_to_binary_tensor(input_recon[data_point_sampled])
        print('output_one_hot', outputs_onehot)
        # print('output dtype:', len(outputs_onehot))  # 查看输出长度   格式.dtype
        outputs_smiles = onehot_to_smiles(outputs_onehot.cpu().detach(), inv_dict)
        print("Input smiles-- ", inputs_smiles)
        print("Output smiles-- ", outputs_smiles)
        inputs_smiles_list.append(inputs_smiles)
        outputs_smiles_list.append(outputs_smiles)
        save_smiles_to_csv(inputs_smiles_list, outputs_smiles_list, r'/mnt/pycharm_project_VAE/smiles.csv')
        '''

        #####################Validation Phase
        epoch_loss_val = 0
        for i, (features, labels) in tqdm(enumerate(val_dataloader)):
            inputs = features.float().to(device)
            targets = labels.float().to(device)
            # print("Val_Inputs Shape:", inputs.shape)
            # inputs = inputs.reshape(batch_size, -1).float()
        # 调整 inputs 的形状，保持 batch 维度，将其它维度展平
            # inputs_flat_val = inputs.permute(0, 2, 1)
            inputs_flat_val = inputs
            input_recon_val, z = model(inputs_flat_val, targets)
            latent_loss_val = latent_loss(model.z_mean, model.z_sigma)
            # 原本是loss = F.binary_cross_entropy(input_recon, inputs, size_average=False) + latent_loss_val
            loss = F.binary_cross_entropy_with_logits(inputs_flat_val.reshape(-1, inputs.size(-1)).float(),
                                                      input_recon.reshape(-1, input_recon_val.size(-1)),
                                                      ) + latent_loss_val
            epoch_loss_val += loss.item()
            print("Validation Loss -- {:.3f}".format(epoch_loss_val))
        print("Validation Loss -- {:.3f}".format(epoch_loss_val / x_val_data_per_epoch))
        scheduler.step(epoch_loss_val)

        ###Add 1 Image per Epoch for Visualisation
        # data_point_sampled = random.randint(0,args.batch_size)
        # add_img(inputs[data_point_sampled], inv_dict, "Original_"+str(epoch))
        # add_img(model(inputs[data_point_sampled:data_point_sampled+1]), inv_dict, "Recon_"+str(epoch))

        checkpoint = {'model': model.state_dict(),
                      'dict': vocab,
                      'inv_dict': inv_dict,
                      }

        # 在尝试保存检查点之前确保目录存在
        save_dir = args.save_loc
        os.makedirs(save_dir, exist_ok=True)

        # Saves when loss is lower than best validation loss till now and all models after 100 epochs
        if epoch_loss_val < best_epoch_loss_val or epoch > 50:
            # if epoch_loss_recon_val < best_epoch_loss_val or epoch > 100:
            torch.save(checkpoint, args.save_loc + '/' + str(epoch) + 'checkpoint.pth')
        # update best epoch loss
        best_epoch_loss_val = min(epoch_loss_val, best_epoch_loss_val)
    # evaluate(model, X_train, vocab, inv_dict)

    # 在合并之前，检查所有z_numpy的形状是否一致
    shapes = [z_numpy.shape for z_numpy in sampled_z_values]
    if len(set(shapes)) != 1:
        raise ValueError(f"Inconsistent shapes found in sampled_z_values: {set(shapes)}")
    # 首先将其转换为 NumPy 数组
    final_z_values = np.concatenate(sampled_z_values, axis=0)
    # 假设 sampled_z_values 是最终的拼接结果
    print(f"Type of sampled_z_values: {type(final_z_values)}")
    print(f"Shape of sampled_z_values: {final_z_values.shape}")

    # 检查是否存在任何 NaN 或无限值
    if np.isnan(final_z_values).any() or np.isinf(final_z_values).any():
        raise ValueError("Data contains NaN or infinite values.")

    # 调用 visualize_latent_space 函数
    visualize_latent_space(final_z_values, use_tsne=False, save_path='pca_visualization.png')
    visualize_latent_space(final_z_values, use_tsne=True, perplexity=20, n_iter=300, save_path='t-SNE_visualization.png')


def evaluate(model, X_train, vocab, inv_dict):
    print("IN EVALUATION PHASE")
    pretrained = torch.load(EVAL_PATH, map_location=lambda storage, loc: storage)
    # torch.load('./Save_Models/189checkpoint.pth',map_location=torch.device('cpu'))
    dataloader = torch.utils.data.DataLoader(X_train,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=2,
                                             drop_last=True)
    for i, data in enumerate(dataloader):
        inputs = data.float().to(args.device)
        input_recon = model(inputs)
        print(i)
        print("Input -- ", onehot_to_smiles_2(inputs[0].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict))
        print("Output -- ", onehot_to_smiles_2(input_recon[0].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict))
        print()


if __name__ == '__main__':
    main()
