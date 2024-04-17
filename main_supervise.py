from model_supervise import *
from utils import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.optim as optim
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import gradcheck
import torch.backends.cudnn as cudnn
from args import get_arguments
from dataset import VAE_Dataset

# torch.backends.cudnn.enabled = False

NUM_EPOCHS = 30
BATCH_SIZE = 64  # 原本是256
LATENT_DIM = 292
RANDOM_SEED = 42
LR = 0.0001
DYN_LR = True
EVAL_PATH = './Save_Models/189checkpoint.pth'

args = get_arguments()
device = args.device


def main():
    data = pd.read_csv(args.data, nrows=2000)  # 取前500行，总共50000行

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
        input_channels = 120
        enc = Conv_Encoder(vocab_size, input_channels=input_channels).to(device)
        dec = GRU_Decoder(vocab_size, args.latent_dim, num_layers=3).to(device)  # 增加了, num_layers=3
        model = Molecule_VAE(enc, dec, device, args.latent_dim).to(device)
        model.get_num_params()

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
    #                                          num_workers=6,
    #                                          drop_last=True)
    # 下面的dataloader中用的dataset加入了性能标签数据label
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=3,
                                             drop_last=True)  # 原本的num_worker=6

    # val_dataloader = torch.utils.data.DataLoader(X_test,
    #                                              batch_size=args.batch_size,
    #                                              shuffle=True,
    #                                              num_workers=6,
    #                                              drop_last=True)
    # 下面的dataloader中用的dataset加入了性能标签数据label
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=3,
                                                 drop_last=True)

    best_epoch_loss_val = 100000
    x_train_data_per_epoch = X_train.shape[0] - X_train.shape[0] % args.batch_size
    x_val_data_per_epoch = X_test.shape[0] - X_test.shape[0] % args.batch_size
    print("Div Quantities", x_train_data_per_epoch, x_val_data_per_epoch)
    print()
    print("###########################################################################")
    # 新增：创建两个列表，用于存储每个step过程的loss值和每个epoch的loss
    step_losses = []
    all_losses = []
    # 新增：创建两个列表，用于存储输入输出smiles生成结果
    inputs_smiles_list = []
    outputs_smiles_list = []

    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Epoch -- {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(epoch))

        for i, (data, label) in tqdm(enumerate(dataloader)):
            inputs = data.float().to(device)
            print("Inputs Shape:", inputs.shape)  # torch.Size([512, 120, 101])
            # 调整 inputs 的形状，保持 batch 维度，将其它维度展平
            # inputs_flat = inputs.permute(0, 2, 1)  # torch.Size([512, 101, 120])
            # inputs_flat = inputs.view(-1, inputs.size(-1))
            inputs_flat = inputs.requires_grad_(True)
            # 原本是inputs = inputs.reshape(batch_size, -1).float()
            optimizer.zero_grad()

            input_recon, outputs_label = model(inputs_flat, label)
            # 检查维度情况打印以下①②③④
            # print("Outputs Shape:", input_recon.shape)  # ①
            # print('output:', input_recon)  # 打印输出张量  # ②
            latent_loss_val = latent_loss(model.z_mean, model.z_sigma)
            print('latent_loss_val:', latent_loss_val)
            # 原本是loss = F.binary_cross_entropy(input_recon, inputs, size_average=False) + latent_loss_val
            # loss = F.binary_cross_entropy_with_logits(inputs_flat.reshape(-1, inputs_flat.size(-1)).float(),
            #                                           input_recon.reshape(-1, input_recon.size(-1)),) + latent_loss_val
            # print('shuruweidu:', input_recon.shape)  # ③
            # print('shuchuweidu:', label.shape)  # ④
            label = torch.unsqueeze(label, 1) # 将label插入dim=1维度，使其匹配outputs_label的(b，1)维度
            loss = F.l1_loss(outputs_label, label)
            #  F.binary_cross_entropy适用于二分类问题，尝试换cross_entropy
            print('loss:', loss)

            # 记录每个step的loss
            step_losses.append(loss.detach().numpy())

            loss.backward()
            # torch.nn.utils.clip_grad_norm_ 函数用于对模型的所有参数进行梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            for name, param in model.named_parameters():
                print(f'Parameter: {name}, Gradient: {param.grad}')
            # 检查梯度
            # for name, param in model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
                    # print(f'Gradient norm of {name}: {param.grad.norm().item()}')
            # gradcheck(model, inputs, eps=1e-6, atol=1e-4)  # 也是检查梯度的工具

            optimizer.step()
            epoch_loss += loss.item()

        # 记录每个 epoch 的 loss
        all_losses.append((epoch_loss / x_train_data_per_epoch))
        # 保存 loss 列表到指定位置
        save_loss_to_csv(step_losses, r'/mnt/pycharm_project_VAE/step_loss.csv')
        save_loss_to_csv(all_losses, r'/mnt/pycharm_project_VAE/epoch_loss.csv')

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

        for i in range(5):  # 取决于想打印前几列
            inputs_smiles = onehot_to_smiles(inputs[i].cpu().detach(), inv_dict)
            outputs_onehot = convert_to_binary_tensor(input_recon[i])
            outputs_smiles = onehot_to_smiles(outputs_onehot.cpu().detach(), inv_dict)
            print("Input smiles-- ", inputs_smiles)
            print("Output smiles-- ", outputs_smiles)
            inputs_smiles_list.append(inputs_smiles)
            outputs_smiles_list.append(outputs_smiles)
            save_smiles_to_csv(inputs_smiles_list, outputs_smiles_list, r'/mnt/pycharm_project_VAE/smiles.csv')

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
        model.eval()
        epoch_loss_val = 0
        for i, (data, label) in enumerate(val_dataloader):
            inputs = data.float().to(device)
            # print("Val_Inputs Shape:", inputs.shape)  # 查看输入维度([batchsize, 120, 110])
            # inputs = inputs.reshape(batch_size, -1).float()
            # 调整 inputs 的形状，保持 batch 维度，将其它维度展平
            # inputs_flat_val = inputs.permute(0, 2, 1)
            inputs_flat_val = inputs
            input_recon_val, outputs_label_val = model(inputs_flat_val, label)
            latent_loss_val = latent_loss(model.z_mean, model.z_sigma)
            # 原本是loss = F.binary_cross_entropy(input_recon, inputs, size_average=False) + latent_loss_val
            # loss = F.binary_cross_entropy_with_logits(inputs_flat_val.reshape(-1, inputs.size(-1)).float(),
            #                                           input_recon.reshape(-1, input_recon_val.size(-1)),
            #                                           ) + latent_loss_val
            loss = F.l1_loss(outputs_label_val, label)
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
        print("Input -- ", onehot_to_smiles(inputs[0].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict))
        print("Output -- ", onehot_to_smiles(input_recon[0].reshape(1, 120, len(vocab)).cpu().detach(), inv_dict))
        print()


if __name__ == '__main__':
    main()
