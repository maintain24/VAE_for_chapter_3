import random
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from args import get_arguments

args = get_arguments()


class Conv_Encoder(nn.Module):
    def __init__(self, vocab_size, input_channels, output_channels=9):
        super(Conv_Encoder, self).__init__()

        self.conv_1 = nn.Conv1d(input_channels, 64, kernel_size=9, padding=4)  # 原本是nn.Conv1d(120, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(64, 120, kernel_size=9, padding=4)  # 原本是nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(120, output_channels, kernel_size=11, padding=5)  # 原本是nn.Conv1d(9, 10, kernel_size=11)
        # vocab_len = 101
        self.fc_1 = nn.Linear(output_channels * (vocab_size + 1), 435 + 1)  # 原本是nn.Linear(10 * (vocab_len - 26), 435)
        self.relu = nn.LeakyReLU(0.01)  # 适当设置负斜率 # 原本是nn.ReLU()
        self.vocab_size = vocab_size  # 添加初始化解决了'Conv_Encoder' object has no attribute 'vocab_len'

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        # print('conv_3_x.shape', x.shape)  # [512, 9, 120]
        x = x.reshape(batch_size, -1)  # [512, 1080]
        x = self.relu(self.fc_1(x))
        return x


class GRU_Decoder(nn.Module):
    def __init__(self, vocab_size, latent_dim, num_layers=1):  # 原本没有，增加了num_layers=3
        super(GRU_Decoder, self).__init__()
        self.fc_1 = nn.Linear(292, 292)
        self.embedding = nn.Embedding(vocab_size, latent_dim)  # 添加嵌入层
        self.gru = nn.GRU(latent_dim, latent_dim, num_layers, batch_first=True)  # 原来的维度是nn.Linear(501, vocab_size)
        self.fc_2 = nn.Linear(latent_dim, 1)  # 原来的维度是nn.Linear(501, vocab_size)
        self.relu = nn.ReLU()  # GRU用不到ReLU激活
        self.softmax = nn.Softmax()
        self.num_layers = num_layers  # 相应对num_layer进行初始化
        self.vocab_size = vocab_size

    def forward(self, z, hidden):
        batch_size = z.shape[0]
        z = torch.clamp(z, min=0, max=self.vocab_size - 1).long()
        z = self.embedding(z.long()).unsqueeze(1)
        z_out, hidden = self.gru(z, hidden)
        # z_out = self.relu(z_out)
        z_out = z_out.contiguous().reshape(-1, z_out.shape[-1])
        x_recon = F.softmax(self.fc_2(z_out), dim=1) + 1e-8  # 添加一个小的 epsilon，以防止数值稳定性问题
        # x_recon = self.fc_2(z_out)  # 原本是self.fc_1(z_out)
        x_recon = x_recon.contiguous().reshape(batch_size, -1, x_recon.shape[-1])

        return x_recon, hidden


class Molecule_VAE(nn.Module):
    def __init__(self, encoder, decoder, device, latent_dim, label_dim=1, teacher_forcing_ratio=0.5):
        super(Molecule_VAE, self).__init__()

        self.encoder = encoder.to(device)
        self.encoder.apply(init_weights_2)  # 权重初始化可能会删除Conv_Encoder的属性
        # 可在init_weight和init_weight_2之间选择初始化函数

        self.decoder = decoder.to(device)
        self.decoder.apply(init_weights_2)  # 可能会删除Conv_Encoder的属性
        # 可在init_weight和init_weight_2之间选择初始化函数

        # self.latent_dim = latent_dim

        # self.hidden_dim_2 = self.encoder.hidden_dim_2
        # self.hidden_dim = self.encoder.hidden_dim
        self.relu = nn.LeakyReLU()  # 原本是nn.ReLU() LeakyReLU有助于缓解梯度消失
        self.device = device

        self._enc_mu = nn.Linear(435 + label_dim, latent_dim)  # + label_dim
        self._enc_log_sigma = nn.Linear(435 + label_dim, latent_dim)  # + label_dim
        self.label_dim = label_dim
        # 添加teacher_forcing_ratio初始化
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def _sample_latent(self, h_enc):
        """Return the latent normal sample z ~ N(mu, sigma^2)"""
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(0.5 * log_sigma)

        eps = torch.randn_like(sigma).float().to(self.device)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * eps  # Reparameterization trick

    def forward_decoder(self, Z, x):
        """Autoregressive Forward Pass through decoder"""
        trg_len = x.shape[1]  # 原本是x.shape[1]
        # print('x.shape[1]:', x.shape[1])  # 120
        # print('x.shape[2]:', x.shape[2])  # 48
        batch_size = Z.shape[0]

        # print(f"self.encoder: {self.encoder}")
        # print(f"self.encoder.vocab_len: {self.encoder.vocab_len}")
        # print(f"self.decoder: {self.decoder}")
        # print(f"self.decoder.num_layers: {self.decoder.num_layers}")
        outputs = torch.zeros(batch_size, trg_len, self.encoder.vocab_size).to(self.device)  # 原本是(batch_size, trg_len, 120)
        outputs[:, 0, 2] = 1  # Intial Output is <STR> Token

        hidden = Z.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)

        input = torch.ones(batch_size).to(self.device) * 2

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            # hidden [1(seq_len), batch_size, dec_hidden]
            # output [1(seq_len), batch_size, vocab_size(dec)]

            output = output.squeeze(0)
            outputs[:, t:t + 1, :] = output
            labels = output.argmax(2).float()
            labels.requires_grad = True

            # Teacher Forcing
            if random.random() < self.teacher_forcing_ratio:
                input = x[:, t, :].argmax(1)

            else:
                input = labels.squeeze(1).detach()  # If this detach is left out the computational graph is retained.
        return outputs, labels

    def forward(self, x, label):
        """Forward Function which passes the data through entire model"""
        # x_with_labels = torch.cat([x, label], dim=1)
        # labels_expanded = label.unsqueeze(2).repeat(1, 1, x.size(2))  # 扩展标签的维度以匹配输入数据
        # 对label（[30]）的维度升维dim=1、2，并扩展dim=1处为x.size(1)，即行数相等，标签添加在最后一列
        labels_expanded = label.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1).float()  # x.size(2)
        x_with_labels = torch.cat([x, labels_expanded], dim=2).float()
        self.h_enc = self.encoder(x_with_labels)
        z = self._sample_latent(self.h_enc)

        recon_x, labels = self.forward_decoder(z, x)
        return recon_x, labels

    def get_num_params(self):
        """Returns the number of Params in all modules"""
        print("Encoder--", sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))
        print("Decoder--", sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))
        print("Total--", sum(p.numel() for p in self.parameters() if p.requires_grad))


def latent_loss(z_mean, z_stddev):
    """Latent Loss used in VAE Model"""
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    # 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
    # 0.5 * torch.mean(mean_sq + z_stddev - torch.log(z_stddev) - 1)
    return -0.5 * torch.mean(torch.log(z_stddev) + 1 - mean_sq - z_stddev)


def init_weights(m):
    """Initialize weights based on type of layer"""
    if type(m) == nn.Conv1d:
        init.normal_(m.weight.data)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Linear:
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
    if type(m) == nn.GRU:
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data, gain=0.01)
            else:
                init.normal_(param.data, mean=0, std=0.01)


# 上一种初始化函数init_weight出现了梯度爆炸或者梯度消失的情况
# 下面的初始化函数init_weight是一种通用初始化方法，适用于线性层和卷积层
def init_weights_2(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

    ##############################################################################################


#######Only Fully Connected Layers in Encoder Decoder (Baseline)##############################
##############################################################################################

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim_2):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim_2)
        self.hidden_dim = hidden_dim
        self.hidden_dim_2 = hidden_dim_2

    def forward(self, x):
        z = F.relu(self.linear2(F.relu(self.linear1(x))))
        return z


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim_2):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(hidden_dim_2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        x = F.relu(self.linear2(F.relu(self.linear1(z))))
        return x


class VAE(nn.Module):

    def __init__(self, encoder, decoder, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

        self.hidden_dim_2 = self.encoder.hidden_dim_2
        self.hidden_dim = self.encoder.hidden_dim

        self._enc_mu = torch.nn.Linear(self.hidden_dim_2, self.latent_dim)
        self._enc_log_sigma = torch.nn.Linear(self.hidden_dim_2, self.latent_dim)

    def _sample_latent(self, h_enc):
        """
		Return the latent normal sample z ~ N(mu, sigma^2)
		"""
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)

        std_z = torch.randn(sigma.size()).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * std_z  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        recon_x = self.decoder(z)
        return recon_x

    def get_num_params(self):
        print("Encoder--", sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))
        print("Decoder--", sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))
        print("Total--", sum(p.numel() for p in self.parameters() if p.requires_grad))


if __name__ == '__main__':
    args = get_arguments()

    print("Checking Normal VAE")
    vocab_size = 71
    input_dim = 120 * vocab_size
    hidden_dim = 200
    hidden_2 = 120
    latent = 60

    enc = Encoder(input_dim, hidden_dim, hidden_2)
    dec = Decoder(input_dim, hidden_dim, latent)
    vae = VAE(enc, dec, latent)
    vae.get_num_params()

    criterion = nn.MSELoss()

    ex_input = torch.randn(1, 120, 71)
    ex_input = ex_input.reshape(1, -1)

    output = vae(ex_input)
    print("MSE LOSS", latent_loss(vae.z_mean, vae.z_sigma) + criterion(ex_input, output))
    print("Input Shape", ex_input.shape)
    print("Output Shape", output.shape)

    print("#######################################################################################")
    print("Checking Molecule VAE")

    enc = Conv_Encoder(input_channels=101, vocab_len = vocab_size)
    dec = GRU_Decoder(vocab_size)

    # device = 'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu'
    device = args.device
    model = Molecule_VAE(enc, dec, device)

    ex_input = torch.randn(1, 120, 71)
    model.get_num_params()
    output = model(ex_input)
    print("MSE LOSS", latent_loss(vae.z_mean, vae.z_sigma) + criterion(ex_input, output))
    print("Input Shape", ex_input.shape)
    print("Output Shape", output.shape)
############################################################################################
