import random
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from args import get_arguments

args = get_arguments()


class Conv_Encoder(nn.Module):
    def __init__(self, vocab_size, input_channels, output_channels=290):  # 原本output_channels=9
        super(Conv_Encoder, self).__init__()

        self.conv_1 = nn.Conv1d(input_channels, 64, kernel_size=15, padding=7)  # 原本是nn.Conv1d(120, 9, kernel_size=9)  # padding=4
        self.conv_2 = nn.Conv1d(64, 120, kernel_size=9, padding=4)  # 原本是nn.Conv1d(9, 9, kernel_size=9)  # padding=4
        self.conv_3 = nn.Conv1d(120, output_channels, kernel_size=11, padding=5)  # 原本是nn.Conv1d(9, 10, kernel_size=11)  # padding=5
        # vocab_len = 101
        self.fc_1 = nn.Linear(output_channels * vocab_size, 435)  # 原本是nn.Linear(10 * (vocab_len - 26), 435)
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


class GRU_Encoder(nn.Module):
    def __init__(self, vocab_size, input_channels, hidden_size=64, num_layers=3, output_channels=435):
        super(GRU_Encoder, self).__init__()

        self.gru = nn.GRU(input_channels, hidden_size, num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, output_channels)
        self.relu = nn.LeakyReLU(0.01)
        self.vocab_size = vocab_size

    def forward(self, x):
        batch_size = x.shape[0]
        x, _ = self.gru(x)
        x = self.relu(x[:, -1, :])
        x = self.fc_1(x)
        print('x.shape:', x.shape)
        return x


class GRU_Decoder(nn.Module):
    def __init__(self, vocab_size, latent_dim, num_layers=5):  # 增加了, num_layers=3
        super(GRU_Decoder, self).__init__()
        self.fc_1 = nn.Linear(292, 292)
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, latent_dim)  # 添加嵌入层,将离散的类别或索引映射到实值向量空间
        self.gru = nn.GRU(latent_dim, latent_dim, num_layers, batch_first=True)  # 原来的维度是nn.GRU(501, vocab_size)
        self.fc_2 = nn.Linear(latent_dim, vocab_size)  # 原来的维度是nn.Linear(501, vocab_size)
        self.relu = nn.LeakyReLU(0.01)  # nn.ReLU()
        self.softmax = nn.Softmax()
        self.num_layers = num_layers  # 相应对num_layer进行初始化
        self.vocab_size = vocab_size
        # 新增一个线性层，将潜在向量转换为GRU的隐藏状态大小
        self.latent_to_hidden = nn.Linear(latent_dim, latent_dim * num_layers)

    def forward(self, z, hidden):
        batch_size = z.shape[0]
        z = torch.clamp(z, min=0, max=self.vocab_size - 1).long()  # 限制张量的值在一个指定的范围内
        z = self.embedding(z.long()).unsqueeze(1)
        z_out, hidden = self.gru(z, hidden)
        z_out = z_out.contiguous().reshape(-1, z_out.shape[-1])
        # x_recon = F.softmax(self.fc_2(z_out), dim=1)  # shape([batchsize,110])  # 建议解码器不用显式使用softmax

        x_recon = self.fc_2(z_out)  # 原本是self.fc_1(z_out)
        # 如果维度是shape([batchsize,110])则不需要下面的reshape
        # x_recon = x_recon.contiguous().reshape(batch_size, -1, x_recon.shape[-1])

        return x_recon, hidden


'''GRU_Decoder_2这种方法中，潜在向量 z 用于初始化GRU的隐藏状态。这允许潜在空间直接影响解码过程的每个步骤，而不仅仅是起始步骤。
优点是，这种方法允许更复杂的潜在空间结构影响解码过程，可能更适合捕捉数据的高维特征。
缺点是，需要额外的线性层来调整潜在向量的维度，以匹配GRU的隐藏状态。(仅仅用于推理生成阶段，不能用于训练阶段)'''
class GRU_Decoder_2(nn.Module):
    def __init__(self, vocab_size, latent_dim, gru_hidden_dim, num_layers=3):
        super(GRU_Decoder_2, self).__init__()
        self.latent_dim = latent_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, latent_dim)
        self.gru = nn.GRU(latent_dim, gru_hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(gru_hidden_dim, vocab_size)

        # 线性层用于将潜在向量调整为GRU的隐藏状态维度
        self.latent_to_hidden = nn.Linear(latent_dim, gru_hidden_dim * num_layers)

    def forward(self, z, max_length):
        batch_size = z.size(0)
        hidden = self.latent_to_hidden(z)
        hidden = hidden.view(self.num_layers, batch_size, self.gru_hidden_dim)

        # 初始化输入序列
        inputs = torch.zeros(batch_size, 1, dtype=torch.long).to(z.device)
        outputs = []

        for _ in range(max_length):
            embedded = self.embedding(inputs).squeeze(1)
            output, hidden = self.gru(embedded, hidden)
            out = self.fc_out(output)
            outputs.append(out)
            inputs = out.argmax(2)

        return torch.cat(outputs, dim=1)


class Molecule_VAE(nn.Module):
    def __init__(self, encoder, decoder, device, latent_dim, teacher_forcing_ratio=1):
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
        self.relu = nn.ReLU()
        self.device = device

        self._enc_mu = nn.Linear(435, latent_dim)
        self._enc_log_sigma = nn.Linear(435, latent_dim)
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
        # print('x.shape[1]:', x.shape[1])  # 120 设定维度
        # print('x.shape[2]:', x.shape[2])  # 48 实际维度
        batch_size = Z.shape[0]  # 确实是32

        outputs = torch.zeros(batch_size, trg_len, self.encoder.vocab_size).to(self.device)  # 原本是(batch_size, trg_len, 120)
        outputs[:, 0, 2] = 1  # Intial Output is <STR> Token
        # outputs[:, 0, :] = F.softmax(outputs[:, 0, :], dim=1)  # 莫名其妙

        hidden = Z.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)  # hidden.shape([3,32,290])
        # hidden = torch.randn(self.decoder.num_layers, batch_size, self.decoder.latent_dim).to(self.device)
        # 将潜在空间改成随机初始化仍然得到高度相似的输出output

        input = torch.ones(batch_size).to(self.device) * 2

        '''GRU_decoder_2()需要重新定义z和max_length'''
        # 准备潜在向量
        # z = torch.randn(batch_size, args.latent_dim)
        # 使用解码器生成数据
        # max_length = 120
        # generated_sequence = self.decoder(z, max_length)

        for t in range(trg_len):  # (1,120)
            output, hidden = self.decoder(input, hidden)
            # hidden [1(seq_len), batch_size, dec_hidden]  # 这是RNN的常见结果维度
            # output [1(seq_len), batch_size, vocab_size(dec)]

            # output = output.squeeze(0)  # 删除第0维，形状从 [1，32, 110] 改为 [32, 110]
            output_expanded = output.unsqueeze(1)  # 将 output 形状从 [32, 110] 改为 [32, 1, 110]
            outputs[:, t:t + 1, :] = output_expanded
            top1 = output.argmax(1)  # 原本是argmax（2）

            # Teacher Forcing
            if random.random() < self.teacher_forcing_ratio:
                # 假设x张量的第二个维度（通常是词汇大小的维度）表示每个时间步的模型输出概率分布
                input = x[:, t, :].argmax(1)  # 为什么不是argmax(2)？

            else:
                # input = top1.squeeze(1).detach()  # If this detach is left out the computational graph is retained.
                input = top1.detach()
                # 这个detach可能是导致梯度消失的真凶
        return outputs

    def forward(self, x):
        """Forward Function which passes the data through entire model"""
        self.h_enc = self.encoder(x)
        z = self._sample_latent(self.h_enc)
        # print('z:', z)

        recon_x = self.forward_decoder(z, x)
        return recon_x

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
    # return -0.5 * torch.mean(torch.log(z_stddev) + 1 - mean_sq - z_stddev)
    return -0.5 * torch.mean(1 + 2 * torch.log(z_stddev) - z_mean.pow(2) - z_stddev.pow(2))


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
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


# 上一种初始化函数init_weight出现了梯度爆炸或者梯度消失的情况,loss为NaN
# 下面的初始化函数init_weight_2是一种通用初始化方法，适用于线性层和卷积层
def init_weights_2(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

# 下面的初始化函数init_weight_3更有针对性，但结果也是cccccccccc
# 对于使用 ReLU 激活函数的 nn.Conv1d 和 nn.Linear 层，使用 He 初始化。
# 对于 nn.GRU 层，使用正交初始化，并将偏置项初始化为零。
# 对于 nn.Conv2d 层，使用 Xavier 初始化，这在使用 Sigmoid 或 Tanh 激活函数时是一个好的选择。
def init_weights_3(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        # 使用 He 初始化 (适用于 ReLU 激活函数)
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.GRU):
        # 对 GRU 层使用正交初始化
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                # 初始化偏置项
                init.zeros_(param.data)
    elif isinstance(m, nn.Conv2d):
        # 对卷积层使用 Xavier 初始化 (如果使用其他激活函数如 Sigmoid 或 Tanh)
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

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
