# -*- coding:utf-8 -*-
import argparse
import torch

NUM_EPOCHS = 30
BATCH_SIZE = 64  # 原本是256
LATENT_DIM = 292
RANDOM_SEED = 42
LR = 0.0001
DYN_LR = True
EVAL_PATH = './Save_Models/189checkpoint.pth'


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular VAE network')
    parser.add_argument('data', type=str, help='Path to the dataset and name')

    parser.add_argument('val_data', type=str, help='Path to the Validation dataset and name')

    parser.add_argument('save_loc', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--model_type', type=str, help='Can Train either Molecular VAE Arch or Vanilla FC VAE',
                            default='mol_vae')
    # parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
    #                     help='Dimensionality of the latent variable.')
    parser.add_argument('--latent_dim', type=int, metavar='N',
                        help='Dimensionality of the latent variable.')  # 删掉了默认default设置
    # parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
    #                     help='Number of samples to process per minibatch during training.')
    parser.add_argument('--batch_size', type=int, metavar='N',
                        help='Number of samples to process per minibatch during training.')  # 删掉了默认default设置
    parser.add_argument('--lr', type=float, metavar='N', default=LR,
                        help='Learning Rate for training.')
    # parser.add_argument('--gpu', type=int, metavar='N', default=0, help='which GPU to use')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()

    return args