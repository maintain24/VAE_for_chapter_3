import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler

"""
夏大佬的vae模型代码，用于多模态数据降维
"""

# Class components
class DownSample(nn.Module):
    """
    SingleConv1D module + MaxPool
    The output dimension = input dimension // down_ratio
    down_ratio是用于控制输出维度与输入维度的比例关系
    这段代码定义了一个下采样模块，由一个一维卷积层（SingleConv1D）、一个最大池化层和一个 dropout 层组成。
    这个模块的作用是将输入张量的维度降低，从而提取输入张量的局部特征。
    """

    def __init__(self, input_chan_num, output_chan_num, down_ratio, kernel_size=9, norm_layer=nn.InstanceNorm1d,
                 leaky_slope=0.2, dropout_p=0):
        """
        Construct a downsampling block
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor 是输入张量的通道数；
            output_chan_num (int) -- the number of channels of the output tensor 是输出张量的通道数
            down_ratio (int)      -- the kernel size and stride of the MaxPool1d layer,下采样因子，控制输出张量的维度与输入张量的维度之比；
            kernel_size (int)     -- the kernel size of the DoubleConv1D block, 卷积核大小，默认为 9
            norm_layer            -- normalization layer, 归一化层，默认为一维实例归一化，这是什么意思
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function 激活函数的负斜率，默认为 0.2；
            dropout_p (float)     -- probability of an element to be zeroed in a dropout layer dropout 层的丢弃概率，默认为 0
        """
        super(DownSample, self).__init__()
        self.down_sample = nn.Sequential(
            SingleConv1D(input_chan_num, output_chan_num, kernel_size, norm_layer, leaky_slope),
            nn.MaxPool1d(down_ratio),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return self.down_sample(x)


class UpSample(nn.Module):
    """
    这段代码定义了一个上采样模块，由一个转置卷积层（ConvTranspose1d）和一个一维卷积层（SingleConv1D）组成。这个模块的作用是将输入张量的维度增加，
    从而恢复输入张量的细节特征。？ 这是叫上采样，下采样还是encoder, decoder呢？
    ConvTranspose1d + SingleConv1D
    The output dimension = input dimension * ratio
    """

    def __init__(self, input_chan_num, output_chan_num, up_ratio, kernel_size=9, norm_layer=nn.BatchNorm1d,
                 leaky_slope=0.2, dropout_p=0, attention=True):
        """
        Construct a upsampling block
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor (the tensor from get from the last layer, not the tensor from the skip-connection mechanism)
            output_chan_num (int) -- the number of channels of the output tensor
            up_ratio (int)        -- the kernel size and stride of the ConvTranspose1d layer
            kernel_size (int)     -- the kernel size of the DoubleConv1D block
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)     -- probability of an element to be zeroed in a dropout layer
            activation (bool)     -- need activation or not
        具体来说，如果 attention 为真，则使用包含 dropout 层、转置卷积层和一维卷积层的 up_sample 序列（nn.Sequential）进行上采样；否则，使用包含 dropout 层、
        转置卷积层和一维卷积层（不包含激活函数）的 up_sample_no_relu 序列进行上采样。
        输出张量的维度将被控制为输入张量维度乘以上采样因子 up_ratio。最终将输出张量作为模块的输出。
        """
        super(UpSample, self).__init__()
        self.attention = attention
        self.up_sample = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.ConvTranspose1d(input_chan_num, input_chan_num, kernel_size=up_ratio, stride=up_ratio),
            SingleConv1D(input_chan_num, output_chan_num, kernel_size, norm_layer, leaky_slope)
        )
        self.up_sample_no_relu = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.ConvTranspose1d(input_chan_num, input_chan_num, kernel_size=up_ratio, stride=up_ratio),
            nn.Conv1d(input_chan_num, output_chan_num, kernel_size=kernel_size, padding=kernel_size // 2)
        )
        '''
        nn.ConvTranspose1d是1D反卷积层，用于将输入特征张量进行上采样操作。kernel_size参数指定了反卷积层的卷积核大小，
        stride参数指定了反卷积层的步长大小。在上采样模块中，反卷积层的kernel_size和stride均为up_ratio，这意味着反卷积层将输入特征张量的长度增加了up_ratio倍。
        反卷积层的作用类似于传统的插值操作，但是它可以通过学习参数来更好地适应数据分布，从而提高上采样的效果。
        '''

    def forward(self, x):
        if self.attention:
            return self.up_sample(x)
        else:
            return self.up_sample_no_relu(x)


class OutputConv(nn.Module):
    """
    Output convolution layer
    """

    def __init__(self, input_chan_num, output_chan_num):
        """
        Construct the output convolution layer
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor
            output_chan_num (int) -- the number of channels of the output omics data
        """
        super(OutputConv, self).__init__()
        self.output_conv = nn.Sequential(
            nn.Conv1d(input_chan_num, output_chan_num, kernel_size=1),
        )
        '''
        对于1D卷积层来说，一个kernel_size=1的卷积操作目的主要是进行通道数变换，而不是进行空间上的卷积操作
        '''

    def forward(self, x):
        return self.output_conv(x)


class SingleConv1D(nn.Module):
    """
    Convolution1D => Norm1D => LeakyReLU
    The omics data dimension keep the same during this process
    这段代码定义了一个包含一个1D卷积层，一个标准化层和一个LeakyReLU激活函数的单卷积块(SingleConv1D)，它用于对输入的1D张量进行特征提取
    """

    def __init__(self, input_chan_num, output_chan_num, kernel_size=9, norm_layer=nn.InstanceNorm1d, leaky_slope=0.2):
        """
        一个1D卷积层，一个标准化层和一个LeakyReLU激活函数的单卷积块(SingleConv1D)，它用于对输入的1D张量进行特征提取。
        在初始化时，需要指定输入通道数(input_chan_num)、输出通道数(output_chan_num)、卷积核大小(kernel_size)、标准化层(norm_layer)
        和LeakyReLU激活函数的负斜率(leaky_slope)。
        Construct a single convolution block
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor, 输入通道
            output_chan_num (int) -- the number of channels of the output tensor， 输出通道
            kernel_size (int)     -- the kernel size of the convolution layer
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
        """
        super(SingleConv1D, self).__init__()

        # Only if the norm method is instance norm we use bias for the corresponding conv layer
        if type(norm_layer) == functools.partial: # 判读标准化层(norm_layer)的类型是否为偏函数
            use_bias = norm_layer.func == nn.InstanceNorm1d # 如果是偏函数，则判断该偏函数的函数名是否为nn.InstanceNorm1d，如果是，则设置卷积层(nn.Conv1d)的偏置项(bias)为True，否则为False
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        self.single_conv_1d = nn.Sequential(
            nn.Conv1d(input_chan_num, output_chan_num, kernel_size=kernel_size, padding=kernel_size // 2,
                      bias=use_bias),
            norm_layer(output_chan_num),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        )

    def forward(self, x):
        return self.single_conv_1d(x)


class FCBlock(nn.Module):
    """
    Linear => Norm1D => LeakyReLU
    这段代码定义了一个全连接块（FCBlock），由一个线性层（nn.Linear）、一个归一化层和一个激活函数组成。
    这个块的作用是将输入张量通过线性变换、归一化和激活函数变换，得到输出张量，用于在神经网络中实现特征提取和分类等任务。
    """
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, activation=True, normalization=True, activation_name='LeakyReLU'):
        """
        Construct a fully-connected block
        定义一个全连接的block
        Parameters:
            input_dim (int)         -- the dimension of the input tensor
            output_dim (int)        -- the dimension of the output tensor
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            activation (bool)       -- need activation or not
            normalization (bool)    -- need normalization or not
            activation_name (str)   -- name of the activation function used in the FC block
        """
        super(FCBlock, self).__init__()
        # Linear
        self.fc_block = [nn.Linear(input_dim, output_dim)] # 添加一个线性变换层
        # Norm ,定义一个归一化层都这么多逻辑
        if normalization: # 如果参数 normalization 为 True，表示需要在全连接层后添加归一化层
            # FC block doesn't support InstanceNorm1d
            '''
            由于全连接层不支持 InstanceNorm1d 归一化层，因此在代码中首先判断当前使用的归一化层是否为 InstanceNorm1d。
            如果是，则将其替换为支持全连接层的 BatchNorm1d 归一化层;
            这是因为 InstanceNorm1d 归一化层主要用于对图像数据进行归一化，而 BatchNorm1d 归一化层则更适合于全连接层的数据。
            '''
            if isinstance(norm_layer, functools.partial) and norm_layer.func == nn.InstanceNorm1d:
                norm_layer = nn.BatchNorm1d
            self.fc_block.append(norm_layer(output_dim))
        # Dropout,定义一个dropout层，对输出向量随机失活，防止过拟合。
        if 0 < dropout_p <= 1:
            self.fc_block.append(nn.Dropout(p=dropout_p))
        # LeakyReLU,定义一个激活函数层，用于对输出向量进行非线性变换，增加模型的表达能力。
        if activation:
            if activation_name.lower() == 'leakyrelu':
                self.fc_block.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
            elif activation_name.lower() == 'tanh':
                self.fc_block.append(nn.Tanh())
            else:
                raise NotImplementedError('Activation function [%s] is not implemented' % activation_name)

        self.fc_block = nn.Sequential(*self.fc_block)
        '''
        最终，self.fc_block 就是一个包含全连接层和归一化层的序列，用于作为一个 block 在 PyTorch 模型中使用。
        这个 block 可以用于构建各种深度学习模型，例如神经网络、卷积神经网络等。
        '''

    def forward(self, x):
        y = self.fc_block(x)
        return y


class Flatten(nn.Module):
    '''
    这里的代码实现了一个通用的展平模块，可以适用于不同大小和形状的输入张量。具体来说，forward函数中的x.view(x.size(0), -1)
    实现了将x张量的第1维保留不变，将其他维展平成一维的操作。其中x.size(0)表示输入x张量的第1维大小，-1表示将其他维度全部展平成一维。
    最终展平后的张量作为forward函数的输出返回。
    '''
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(nn.Module):
    '''
    具体来说，这个Unflatten模块的初始化函数__init__()接受两个参数：channel和dim，分别表示重新变形后的张量的通道数和某一维的大小。在forward函数中，
    x.view(x.size(0), self.channel, self.dim)实现了将x张量重新变形成多维张量的操作。其中x.size(0)表示输入x张量的第1维大小，self.channel和
    self.dim分别表示输出张量的通道数和某一维的大小。最终变形后的张量作为forward函数的输出返回。
    '''
    def __init__(self, channel, dim):
        super(Unflatten, self).__init__()
        '''
        在Python中，super()是一个用于调用父类方法的函数。在这个代码中，super(Unflatten, self).init()是在Unflatten,
        类的初始化函数中调用了其父类nn.Module的初始化函数。
        '''
        self.channel = channel
        self.dim = dim

    def forward(self, x):
        return x.view(x.size(0), self.channel, self.dim)
    '''
    具体来说，这个Unflatten模块的初始化函数__init__()接受两个参数：channel和dim，分别表示重新变形后的张量的通道数和某一维的大小。
    在forward函数中，x.view(x.size(0), self.channel, self.dim)实现了将x张量重新变形成多维张量的操作。其中x.size(0)表示输入x张量的第1维大小，
    self.channel和self.dim分别表示输出张量的通道数和某一维的大小。最终变形后的张量作为forward函数的输出返回。
    '''

class Identity(nn.Module):
    '''
    这段代码定义了一个名为Identity的PyTorch模型，它实现了一个简单的恒等映射，即将输入x直接返回输出，没有进行任何操作。
    实现这个模型的目的通常是为了在其他模型中进行连接或跳跃连接（skip connection）时使用，或者用作其他需要输入和输出形状相同的模型的组件。
    常常需要在不同层之间进行连接，以便实现更复杂的模型。例如，在残差网络（ResNet）中，可以使用跳跃连接来保留前一层的特征，以便更好地学习深层特征。
    当前一层的输出x需要与后续层的输出进行加和，这时候就可以使用Identity模型来将x直接传递给后续层，实现跳跃连接。
    '''
    def forward(self, x):
        return x


# Class for VAE
# ConvVae
class ConvVaeABC(nn.Module):
    """
        Defines a one dimensional convolution variational autoencoder for multi-omics dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.InstanceNorm1d, filter_num=8, kernel_size=9, leaky_slope=0.2,
                 dropout_p=0, ratio_1B=16, ratio_2B=16, ratio_1A=4, ratio_2A=4, ratio_1C=2, ratio_2C=2, ratio_3=16,
                 latent_dim=256):
        """
            Construct a one dimensional convolution variational autoencoder for multi-omics dataset
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                filter_num (int)        -- the number of filters in the first convolution layer in the VAE
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(ConvVaeABC, self).__init__()

        A_dim = omics_dims[0]
        B_dim = omics_dims[1]
        C_dim = omics_dims[2]

        hidden_dim_1 = (B_dim // ratio_1B // ratio_2B + A_dim // ratio_1A // ratio_2A + C_dim // ratio_1C // ratio_2C) // ratio_3 * filter_num * 4
        hidden_dim_2 = (hidden_dim_1 // (filter_num * 4) + 1) * filter_num * 4
        self.narrow_B = hidden_dim_2 // (4 * filter_num) * ratio_3 * (B_dim // ratio_1B // ratio_2B) // (
                B_dim // ratio_1B // ratio_2B + A_dim // ratio_1A // ratio_2A + C_dim // ratio_1C // ratio_2C)
        self.narrow_A = hidden_dim_2 // (4 * filter_num) * ratio_3 * (A_dim // ratio_1A // ratio_2A) // (
                B_dim // ratio_1B // ratio_2B + A_dim // ratio_1A // ratio_2A + C_dim // ratio_1C // ratio_2C)
        self.narrow_C = hidden_dim_2 // (4 * filter_num) * ratio_3 * (C_dim // ratio_1C // ratio_2C) // (
                B_dim // ratio_1B // ratio_2B + A_dim // ratio_1A // ratio_2A + C_dim // ratio_1C // ratio_2C)
        self.B_dim = B_dim
        self.A_dim = A_dim
        self.C_dim = C_dim
        '''
    hidden_dim_1的值是三个项的和，每一个项都包含一个输入维度(B_dim, A_dim, C_dim)除以三个比率(ratio_1B, ratio_2B, ratio_1A, ratio_2A, ratio_1C, ratio_2C)，然后再除以ratio_3，乘以filter_num，最后再乘以4。

    hidden_dim_2的值是hidden_dim_1除以filter_num * 4，然后向上取整，最后再乘以filter_num * 4。

    self.narrow_B的值是hidden_dim_2除以4 * filter_num，再乘以ratio_3，再乘以B_dim // ratio_1B // ratio_2B，最后再除以步骤1中三个项的和。

    self.narrow_A的计算方法与self.narrow_B类似，但是使用的是A_dim // ratio_1A // ratio_2A，而不是B_dim // ratio_1B // ratio_2B。

    self.narrow_C的计算方法与self.narrow_B类似，但是使用的是C_dim // ratio_1C // ratio_2C，而不是B_dim // ratio_1B // ratio_2B。
    
    总体上，这段代码似乎是基于输入参数和ratio值进行一些维度压缩或降维的处理。具体这段代码在整个程序或模型中的作用和目的会依赖于上下文。
        '''

        # ENCODER
        # B 1 -> 8
        self.down_sample_1B = DownSample(1, filter_num, down_ratio=ratio_1B, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B 8 -> 16
        self.down_sample_2B = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2B, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # A 1 -> 8
        self.down_sample_1A = DownSample(1, filter_num, down_ratio=ratio_1A, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # A 8 -> 16
        self.down_sample_2A = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2A, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # C 1 -> 8
        self.down_sample_1C = DownSample(1, filter_num, down_ratio=ratio_1C, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # C 8 -> 16
        self.down_sample_2C = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2C, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)

        # 16 -> 32
        self.down_sample_3 = DownSample(filter_num * 2, filter_num * 4, down_ratio=ratio_3, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # Flatten
        self.flatten = Flatten()
        # FC to mean
        self.fc_mean = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                               dropout_p=0, activation=False, normalization=False)
        # FC to log_var
        self.fc_log_var = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                  dropout_p=0, activation=False, normalization=False)

        # DECODER
        # FC from z
        self.fc_z = FCBlock(latent_dim, hidden_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=True)
        # Unflatten
        self.unflatten = Unflatten(filter_num * 4, hidden_dim_2 // (4 * filter_num))
        # 32 -> 16
        self.up_sample_1 = UpSample(filter_num * 4, filter_num * 2, up_ratio=ratio_3, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B 16 -> 8
        self.up_sample_2B = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2B, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # B 8 -> 1
        self.up_sample_3B = UpSample(filter_num, filter_num, up_ratio=ratio_1B, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B Output
        self.output_conv_B = OutputConv(filter_num, 1)

        # A 16 -> 8
        self.up_sample_2A = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2A, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # A 8 -> 1
        self.up_sample_3A = UpSample(filter_num, filter_num, up_ratio=ratio_1A, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # A Output
        self.output_conv_A = OutputConv(filter_num, 1)

        # C 16 -> 8
        self.up_sample_2C = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2C, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # C 8 -> 1
        self.up_sample_3C = UpSample(filter_num, filter_num, up_ratio=ratio_1C, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # C Output
        self.output_conv_C = OutputConv(filter_num, 1)

    def encode(self, x):
        level_2_B = self.down_sample_1B(x[1])
        level_2_A = self.down_sample_1A(x[0])
        level_2_C = self.down_sample_1C(x[2])

        level_3_B = self.down_sample_2B(level_2_B)
        level_3_A = self.down_sample_2A(level_2_A)
        level_3_C = self.down_sample_2C(level_2_C)

        level_3 = torch.cat((level_3_B, level_3_A, level_3_C), 2)

        level_4 = self.down_sample_3(level_3)
        level_4_flatten = self.flatten(level_4)

        latent_mean = self.fc_mean(level_4_flatten)
        latent_log_var = self.fc_log_var(level_4_flatten)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.fc_z(z)
        level_1_unflatten = self.unflatten(level_1)

        level_2 = self.up_sample_1(level_1_unflatten)
        level_2_B = level_2.narrow(2, 0, self.narrow_B)
        level_2_A = level_2.narrow(2, self.narrow_B, self.narrow_A)
        level_2_C = level_2.narrow(2, self.narrow_B+self.narrow_A, self.narrow_C+1)

        '''
        self.narrow_B和self.narrow_A是用来切分level_2张量的，以便在后续的上采样过程中，能够正确地恢复出原始的A和B的形状。具体来说，level_2张量的第2个维度被切成了3个不同的子张量：

        level_2_B：从第0个元素开始，切出长度为self.narrow_B的子张量，用于恢复B的形状。
        level_2_A：从第self.narrow_B个元素开始，切出长度为self.narrow_A的子张量，用于恢复A的形状。
        level_2_C：从第self.narrow_B + self.narrow_A个元素开始，切出长度为self.narrow_C + 1的子张量，用于恢复C的形状。
        这些子张量被送入相应的上采样层进行处理，最终输出恢复后的A、B和C。因此，self.narrow_B和self.narrow_A的值对于恢复原始形状非常重要。
        '''

        level_3_B = self.up_sample_2B(level_2_B)
        level_3_A = self.up_sample_2A(level_2_A)
        level_3_C = self.up_sample_2C(level_2_C)

        level_4_B = self.up_sample_3B(level_3_B)
        level_4_A = self.up_sample_3A(level_3_A)
        level_4_C = self.up_sample_3C(level_3_C)

        output_B = self.output_conv_B(level_4_B)
        output_A = self.output_conv_A(level_4_A)
        output_C = self.output_conv_C(level_4_C)

        recon_B = output_B[:, :, 0:self.B_dim]
        recon_A = output_A[:, :, 0:self.A_dim]
        recon_C = output_C[:, :, 0:self.C_dim]

        return [recon_A, recon_B, recon_C]

    def get_last_encode_layer(self):
        return self.fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class ConvVaeAB(nn.Module):
    """
        Defines a one dimensional convolution variational autoencoder for multi-omics dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.InstanceNorm1d, filter_num=8, kernel_size=9, leaky_slope=0.2,
                 dropout_p=0, ratio_1B=16, ratio_2B=16, ratio_1A=4, ratio_2A=4, ratio_3=16, latent_dim=256):
        """
            Construct a one dimensional convolution variational autoencoder for multi-omics dataset
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                filter_num (int)        -- the number of filters in the first convolution layer in the VAE
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(ConvVaeAB, self).__init__()

        A_dim = omics_dims[0]
        B_dim = omics_dims[1]

        hidden_dim_1 = (B_dim // ratio_1B // ratio_2B + A_dim // ratio_1A // ratio_2A) // ratio_3 * filter_num * 4
        hidden_dim_2 = (hidden_dim_1 // (filter_num * 4) + 2) * filter_num * 4
        self.narrow_B = hidden_dim_2 // (4 * filter_num) * ratio_3 * (B_dim // ratio_1B // ratio_2B) // (
                B_dim // ratio_1B // ratio_2B + A_dim // ratio_1A // ratio_2A)
        self.narrow_A = hidden_dim_2 // (4 * filter_num) * ratio_3 - self.narrow_B
        self.B_dim = B_dim
        self.A_dim = A_dim

        # ENCODER
        # B 1 -> 8
        self.down_sample_1B = DownSample(1, filter_num, down_ratio=ratio_1B, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B 8 -> 16
        self.down_sample_2B = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2B, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # A 1 -> 8
        self.down_sample_1A = DownSample(1, filter_num, down_ratio=ratio_1A, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # A 8 -> 16
        self.down_sample_2A = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2A, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 16 -> 32
        self.down_sample_3 = DownSample(filter_num * 2, filter_num * 4, down_ratio=ratio_3, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # Flatten
        self.flatten = Flatten()
        # FC to mean
        self.fc_mean = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                               dropout_p=0, activation=False, normalization=False)
        # FC to log_var
        self.fc_log_var = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                  dropout_p=0, activation=False, normalization=False)

        # DECODER
        # FC from z
        self.fc_z = FCBlock(latent_dim, hidden_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=True)
        # Unflatten
        self.unflatten = Unflatten(filter_num * 4, hidden_dim_2 // (4 * filter_num))
        # 32 -> 16
        self.up_sample_1 = UpSample(filter_num * 4, filter_num * 2, up_ratio=ratio_3, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B 16 -> 8
        self.up_sample_2B = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2B, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # B 8 -> 1
        self.up_sample_3B = UpSample(filter_num, filter_num, up_ratio=ratio_1B, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B Output
        self.output_conv_B = OutputConv(filter_num, 1)

        # A 16 -> 8
        self.up_sample_2A = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2A, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # A 8 -> 1
        self.up_sample_3A = UpSample(filter_num, filter_num, up_ratio=ratio_1A, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # A Output
        self.output_conv_A = OutputConv(filter_num, 1)

    def encode(self, x):
        level_2_B = self.down_sample_1B(x[1])
        level_2_A = self.down_sample_1A(x[0])
        level_3_B = self.down_sample_2B(level_2_B)
        level_3_A = self.down_sample_2A(level_2_A)
        level_3 = torch.cat((level_3_B, level_3_A), 2)

        level_4 = self.down_sample_3(level_3)
        level_4_flatten = self.flatten(level_4)

        latent_mean = self.fc_mean(level_4_flatten)
        latent_log_var = self.fc_log_var(level_4_flatten)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.fc_z(z)
        level_1_unflatten = self.unflatten(level_1)

        level_2 = self.up_sample_1(level_1_unflatten)
        level_2_B = level_2.narrow(2, 0, self.narrow_B)
        level_2_A = level_2.narrow(2, self.narrow_B, self.narrow_A)

        level_3_B = self.up_sample_2B(level_2_B)
        level_3_A = self.up_sample_2A(level_2_A)

        level_4_B = self.up_sample_3B(level_3_B)
        level_4_A = self.up_sample_3A(level_3_A)

        output_B = self.output_conv_B(level_4_B)
        output_A = self.output_conv_A(level_4_A)

        recon_B = output_B[:, :, 0:self.B_dim]
        recon_A = output_A[:, :, 0:self.A_dim]

        return [recon_A, recon_B]

    def get_last_encode_layer(self):
        return self.fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class ConvVaeB(nn.Module):
    """
        Defines a one dimensional convolution variational autoencoder for DNA methylation dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.InstanceNorm1d, filter_num=8, kernel_size=9, leaky_slope=0.2,
                 dropout_p=0, ratio_1B=16, ratio_2B=16, ratio_3=16, latent_dim=256):
        """
            Construct a one dimensional convolution variational autoencoder for DNA methylation dataset
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                filter_num (int)        -- the number of filters in the first convolution layer in the VAE
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(ConvVaeB, self).__init__()

        B_dim = omics_dims[1]

        hidden_dim_1 = B_dim // ratio_1B // ratio_2B // ratio_3 * filter_num * 4
        hidden_dim_2 = (hidden_dim_1 // (filter_num * 4) + 1) * filter_num * 4

        self.B_dim = B_dim

        # ENCODER
        # B 1 -> 8
        self.down_sample_1B = DownSample(1, filter_num, down_ratio=ratio_1B, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B 8 -> 16
        self.down_sample_2B = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2B, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 16 -> 32
        self.down_sample_3 = DownSample(filter_num * 2, filter_num * 4, down_ratio=ratio_3, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # Flatten
        self.flatten = Flatten()
        # FC to mean
        self.fc_mean = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                               dropout_p=0, activation=False, normalization=False)
        # FC to log_var
        self.fc_log_var = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                  dropout_p=0, activation=False, normalization=False)

        # DECODER
        # FC from z
        self.fc_z = FCBlock(latent_dim, hidden_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=True)
        # Unflatten
        self.unflatten = Unflatten(filter_num * 4, hidden_dim_2 // (4 * filter_num))
        # 32 -> 16
        self.up_sample_1 = UpSample(filter_num * 4, filter_num * 2, up_ratio=ratio_3, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B 16 -> 8
        self.up_sample_2B = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2B, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # B 8 -> 1
        self.up_sample_3B = UpSample(filter_num, filter_num, up_ratio=ratio_1B, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # B Output
        self.output_conv_B = OutputConv(filter_num, 1)

    def encode(self, x):
        level_2_B = self.down_sample_1B(x[1])
        level_3_B = self.down_sample_2B(level_2_B)

        level_4 = self.down_sample_3(level_3_B)
        level_4_flatten = self.flatten(level_4)

        latent_mean = self.fc_mean(level_4_flatten)
        latent_log_var = self.fc_log_var(level_4_flatten)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.fc_z(z)
        level_1_unflatten = self.unflatten(level_1)
        level_2 = self.up_sample_1(level_1_unflatten)
        level_3_B = self.up_sample_2B(level_2)
        level_4_B = self.up_sample_3B(level_3_B)
        output_B = self.output_conv_B(level_4_B)
        recon_B = output_B[:, :, 0:self.B_dim]

        return [None, recon_B]

    def get_last_encode_layer(self):
        return self.fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class ConvVaeA(nn.Module):
    """
        Defines a one dimensional convolution variational autoencoder for gene expression dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.InstanceNorm1d, filter_num=8, kernel_size=9, leaky_slope=0.2,
                 dropout_p=0, ratio_1A=4, ratio_2A=4, ratio_3=16, latent_dim=256):
        """
            Construct a one dimensional convolution variational autoencoder for multi-omics dataset
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                filter_num (int)        -- the number of filters in the first convolution layer in the VAE
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(ConvVaeA, self).__init__()

        A_dim = omics_dims[0]

        hidden_dim_1 = A_dim // ratio_1A // ratio_2A // ratio_3 * filter_num * 4
        hidden_dim_2 = (hidden_dim_1 // (filter_num * 4) + 1) * filter_num * 4
        self.A_dim = A_dim

        # ENCODER
        # A 1 -> 8
        self.down_sample_1A = DownSample(1, filter_num, down_ratio=ratio_1A, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # A 8 -> 16
        self.down_sample_2A = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2A, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 16 -> 32
        self.down_sample_3 = DownSample(filter_num * 2, filter_num * 4, down_ratio=ratio_3, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # Flatten
        self.flatten = Flatten()
        # FC to mean
        self.fc_mean = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                               dropout_p=0, activation=False, normalization=False)
        # FC to log_var
        self.fc_log_var = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                  dropout_p=0, activation=False, normalization=False)

        # DECODER
        # FC from z
        self.fc_z = FCBlock(latent_dim, hidden_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=True)
        # Unflatten
        self.unflatten = Unflatten(filter_num * 4, hidden_dim_2 // (4 * filter_num))
        # 32 -> 16
        self.up_sample_1 = UpSample(filter_num * 4, filter_num * 2, up_ratio=ratio_3, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)

        # A 16 -> 8
        self.up_sample_2A = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2A, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # A 8 -> 1
        self.up_sample_3A = UpSample(filter_num, filter_num, up_ratio=ratio_1A, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # A Output
        self.output_conv_A = OutputConv(filter_num, 1)

    def encode(self, x):
        level_2_A = self.down_sample_1A(x[0])
        level_3_A = self.down_sample_2A(level_2_A)
        level_4 = self.down_sample_3(level_3_A)
        level_4_flatten = self.flatten(level_4)

        latent_mean = self.fc_mean(level_4_flatten)
        latent_log_var = self.fc_log_var(level_4_flatten)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.fc_z(z)
        level_1_unflatten = self.unflatten(level_1)

        level_2 = self.up_sample_1(level_1_unflatten)

        level_3_A = self.up_sample_2A(level_2)

        level_4_A = self.up_sample_3A(level_3_A)

        output_A = self.output_conv_A(level_4_A)

        recon_A = output_A[:, :, 0:self.A_dim]

        return [recon_A]

    def get_last_encode_layer(self):
        return self.fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class ConvVaeC(nn.Module):
    """
        Defines a one dimensional convolution variational autoencoder for miRNA expression dataset
        这是一个 PyTorch 模型类 ConvVaeC，它实现了一个一维卷积变分自编码器（Convolutional Variational Autoencoder）用于处理 miRNA 表达数据集。
        该模型可以进行数据压缩和重建，并且可以用于生成新的 miRNA 表达数据。
    """
    def __init__(self, omics_dims, norm_layer=nn.InstanceNorm1d, filter_num=8, kernel_size=9, leaky_slope=0.2,
                 dropout_p=0, ratio_1C=2, ratio_2C=2, ratio_3=16, latent_dim=256):
        """
            Construct a one dimensional convolution variational autoencoder for multi-omics dataset
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                filter_num (int)        -- the number of filters in the first convolution layer in the VAE
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(ConvVaeC, self).__init__()

        C_dim = omics_dims[2]
        '''
        C_dim代表输入数据集的特征维度，也就是每个数据样本的特征数量。在卷积神经网络中，输入数据集通常会被转换为一个四维的张量，
        其四个维度分别是batch size、通道数、高度和宽度。在这里，C_dim代表的是输入数据集的通道数，也就是数据集的特征维度。
        '''
        print(f"打印出c_dim, 看看输出的是不是的四个维度的张量:")
        print(C_dim)
        print("结束！")
        hidden_dim_1 = (C_dim // ratio_1C // ratio_2C) // ratio_3 * filter_num * 4
        hidden_dim_2 = (hidden_dim_1 // (filter_num * 4) + 1) * filter_num * 4
        self.C_dim = C_dim
        '''
        C_dim是输入特征图在通道维度上的大小，而ratio_1C、ratio_2C和ratio_3是应用于特征图沿此维度的三个下采样比率。（这个不make sure）
        1. 特征图通常是一个四维张量，其四个维度分别是：batch size、通道数、高度和宽度。因此，C_dim代表的是输入特征图在通道维度上的大小
        ，也就是特征图的第二个维度。通道数是特征图的第二个维度，而不是第三个维度。
        2. 符号“//”表示整数除法，它与普通的除法“/”不同。在整数除法中，当除数不能整除被除数时，结果将向下取整，得到一个整数结果。
        3. 通道数是卷积神经网络中一个重要的超参数，它指定了在每个卷积层中要使用多少个卷积核（或滤波器）。卷积核在卷积神经网络中起到提取特征的作用，
        每个卷积核会对输入的特征图进行卷积操作，生成一个新的特征图，其中每个像素值都是卷积核与输入特征图上某个位置的乘积之和。
        表达式(C_dim // ratio_1C // ratio_2C) // ratio_3对应于在应用了所有三个下采样操作后的特征图大小。
        '''

        # ENCODER
        # C 1 -> 8
        self.down_sample_1C = DownSample(1, filter_num, down_ratio=ratio_1C, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        '''
        编码器部分的 self.down_sample_1C 是一个卷积层，它的作用是将输入数据的通道数从1升到8。这个操作是为了引入更多的特征信息，以便更好地捕捉输入数据的特征。在卷积神经网络中，
        卷积层的通道数通常对应着特征的数量，它们可以对输入数据进行各种不同的变换和特征提取，以获得更高质量的特征表示。通过增加通道数，模型可以更好地学习输入数据的特征，
        从而提高模型的表现力和泛化性能。
        需要注意的是，通道数的增加不一定会导致模型的性能提高，而是需要根据具体的任务和数据集来确定。在实践中，通常需要进行多轮实验和调整，以找到最优的模型结构和参数配置。
        '''
        # C 8 -> 16
        self.down_sample_2C = DownSample(filter_num, filter_num * 2, down_ratio=ratio_2C, kernel_size=kernel_size,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        '''
        是的，self.down_sample_1C 是一个下采样层，它将输入数据在空间维度上进行降采样，同时增加通道数。在这个层中，输入数据的通道数从1增加到了 filter_num，
        这个 filter_num 是模型中的一个超参数，表示卷积层的滤波器数量。
        '''
        # 16 -> 32
        self.down_sample_3 = DownSample(filter_num * 2, filter_num * 4, down_ratio=ratio_3, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # Flatten
        self.flatten = Flatten()
        # FC to mean
        self.fc_mean = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                               dropout_p=0, activation=False, normalization=False)
        # FC to log_var
        self.fc_log_var = FCBlock(hidden_dim_1, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                  dropout_p=0, activation=False, normalization=False)

        # DECODER
        # FC from z
        self.fc_z = FCBlock(latent_dim, hidden_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=True)
        # Unflatten
        self.unflatten = Unflatten(filter_num * 4, hidden_dim_2 // (4 * filter_num))
        '''
        这个太不make sure了
        具体来说，这段代码中的 Unflatten 层接收一个长度为 filter_num * 4 的一维张量数据，并将其转换为一个形状为
         (batch_size, hidden_dim_2 // (4 * filter_num),  4 * filter_num) 的三维张量数据，
         其中 batch_size 是输入数据的样本数，hidden_dim_2 是网络中的隐藏层维度，
         filter_num 是卷积核的数量。
         这里，hidden_dim_2 // (4 * filter_num) 表示对隐藏层维度进行等分，每个卷积核对应 4 个等分部分。
        '''
        # 32 -> 16
        self.up_sample_1 = UpSample(filter_num * 4, filter_num * 2, up_ratio=ratio_3, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)

        # C 16 -> 8
        self.up_sample_2C = UpSample(filter_num * 2, filter_num, up_ratio=ratio_2C, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # C 8 -> 1
        self.up_sample_3C = UpSample(filter_num, filter_num, up_ratio=ratio_1C, kernel_size=kernel_size,
                                     norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # C Output
        self.output_conv_C = OutputConv(filter_num, 1)

    def encode(self, x):
        level_2_C = self.down_sample_1C(x[2])

        level_3_C = self.down_sample_2C(level_2_C)

        level_4 = self.down_sample_3(level_3_C)
        level_4_flatten = self.flatten(level_4)

        latent_mean = self.fc_mean(level_4_flatten)
        latent_log_var = self.fc_log_var(level_4_flatten)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.fc_z(z)
        level_1_unflatten = self.unflatten(level_1)

        level_2 = self.up_sample_1(level_1_unflatten)

        level_3_C = self.up_sample_2C(level_2)

        level_4_C = self.up_sample_3C(level_3_C)

        output_C = self.output_conv_C(level_4_C)

        recon_C = output_C[:, :, 0:self.C_dim]

        return [None, None, recon_C]

    def get_last_encode_layer(self):
        return self.fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


# FcSepVae
class FcSepVaeABC(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input separated by chromosome
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=128, dim_2B=1024,
                 dim_1A=2048, dim_2A=1024, dim_1C=1024, dim_2C=1024, dim_3=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcSepVaeABC, self).__init__()

        self.A_dim = omics_dims[0]
        self.B_dim = omics_dims[1]
        self.C_dim = omics_dims[2]
        self.dim_1B = dim_1B
        self.dim_2B = dim_2B
        self.dim_2A = dim_2A
        self.dim_2C = dim_2C

        # ENCODER
        # Layer 1
        self.encode_fc_1B_list = nn.ModuleList()
        # for i in range(0, 23):
        #     self.encode_fc_1B_list.append(
        #         FCBlock(self.B_dim_list[i], dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
        #                 activation=True))
        self.encode_fc_1B = FCBlock(self.B_dim, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_1C = FCBlock(self.C_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True) # 修改了dim_1B*23 ---> dim_1B
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2C = FCBlock(dim_1C, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B+dim_2A+dim_2C, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B+dim_2A+dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True) # 这里修改了dim_1B *23 ---> dim_1B
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3C = FCBlock(dim_2C, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 4
        self.decode_fc_4B_list = nn.ModuleList()
        # for i in range(0, 23):
        #     self.decode_fc_4B_list.append(
        #         FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
        #                 activation=False, normalization=False))
        self.decode_fc_4B = FCBlock(dim_1B, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_4A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_4C = FCBlock(dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B_list = []
        # for i in range(0, 23):
        #     level_2_B_list.append(self.encode_fc_1B_list[i](x[1][i]))

        # level_2_B = torch.cat(level_2_B_list, 1)
        level_2_B = self.encode_fc_1B(x[1])
        level_2_A = self.encode_fc_1A(x[0])
        level_2_C = self.encode_fc_1C(x[2])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3_C = self.encode_fc_2C(level_2_C)
        level_3 = torch.cat((level_3_B, level_3_A, level_3_C), 1)

        level_4 = self.encode_fc_3(level_3)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)
        level_2_B = level_2.narrow(1, 0, self.dim_2B)
        level_2_A = level_2.narrow(1, self.dim_2B, self.dim_2A)
        level_2_C = level_2.narrow(1, self.dim_2B+self.dim_2A, self.dim_2C)

        level_3_B = self.decode_fc_3B(level_2_B)
        # level_3_B_list = []
        # for i in range(0, 23):
        #     level_3_B_list.append(level_3_B.narrow(1, self.dim_1B*i, self.dim_1B))
        level_3_A = self.decode_fc_3A(level_2_A)
        level_3_C = self.decode_fc_3C(level_2_C)

        # recon_B_list = []
        # for i in range(0, 23):
        #     recon_B_list.append(self.decode_fc_4B_list[i](level_3_B_list[i]))
        recon_B = self.decode_fc_4B(level_3_B)
        recon_A = self.decode_fc_4A(level_3_A)
        recon_C = self.decode_fc_4C(level_3_C)

        return [recon_A, recon_B, recon_C]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcSepVaeAB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input separated by chromosome
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=128, dim_2B=1024,
                 dim_1A=2048, dim_2A=1024, dim_3=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcSepVaeAB, self).__init__()

        self.A_dim = omics_dims[0]
        self.B_dim_list = omics_dims[1] #  DNA methylation input separated by chromosome
        self.dim_1B = dim_1B
        self.dim_2B = dim_2B
        self.dim_2A = dim_2A

        # ENCODER
        # Layer 1
        self.encode_fc_1B_list = nn.ModuleList()
        for i in range(0, 23):
            # encode_fc_1B_list 包含了多个 FCBlock，每个 FCBlock 将输入的 B 数据压缩为 dim_1B 维度的向量。这里使用了一个 ModuleList 来表示多个 FCBlock，通过循环添加到列表中。B_dim_list 是一个列表，包含了每个 B 数据的维度，共有 23 个。
            self.encode_fc_1B_list.append(
                FCBlock(self.B_dim_list[i], dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 2
        # 其中，encode_fc_2A 是将第一层得到的 A 数据的低维度向量压缩为 dim_2A 维度的向量。dim_1A 表示第一层压缩后的维度，dim_2A 表示第二层压缩后的维度。
        #
        # encode_fc_2B 是将第一层得到的 23 个 B 数据的低维度向量首尾拼接起来，再压缩为 dim_2B 维度的向量。dim_1B 表示第一层压缩后的维度，dim_2B 表示第二层压缩后的维度。
        #
        # 这里仍然使用了 FCBlock 类来表示全连接层，该类包含了全连接层、批标准化、激活函数、dropout 等一系列操作。
        self.encode_fc_2B = FCBlock(dim_1B*23, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B+dim_2A, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # 第二层得到的 A 和 B 低维度向量拼接在一起，压缩为 dim_3 维度的向量。这里使用了 FCBlock 类来表示全连接层，该类包含了全连接层、批标准化、激活函数、dropout 等一系列操作。
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)
        # 第二层得到的 A 和 B 低维度向量拼接在一起，压缩为 dim_3 维度的向量。这里使用了 FCBlock 类来表示全连接层，该类包含了全连接层、批标准化、激活函数、dropout 等一系列操作。
        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B+dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B*23, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 4
        self.decode_fc_4B_list = nn.ModuleList()
        for i in range(0, 23):
            self.decode_fc_4B_list.append(
                FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                        activation=False, normalization=False))
        self.decode_fc_4A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B_list = []
        for i in range(0, 23):
            level_2_B_list.append(self.encode_fc_1B_list[i](x[1][i]))
        level_2_B = torch.cat(level_2_B_list, 1)
        level_2_A = self.encode_fc_1A(x[0])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3 = torch.cat((level_3_B, level_3_A), 1)

        level_4 = self.encode_fc_3(level_3)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)
        level_2_B = level_2.narrow(1, 0, self.dim_2B)
        level_2_A = level_2.narrow(1, self.dim_2B, self.dim_2A)

        level_3_B = self.decode_fc_3B(level_2_B)
        level_3_B_list = []
        for i in range(0, 23):
            level_3_B_list.append(level_3_B.narrow(1, self.dim_1B*i, self.dim_1B))
        level_3_A = self.decode_fc_3A(level_2_A)

        recon_B_list = []
        for i in range(0, 23):
            recon_B_list.append(self.decode_fc_4B_list[i](level_3_B_list[i]))
        recon_A = self.decode_fc_4A(level_3_A)

        return [recon_A, recon_B_list]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcSepVaeB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for DNA methylation dataset
        DNA methylation input separated by chromosome
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=128, dim_2B=1024,
                 dim_3=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcSepVaeB, self).__init__()

        self.B_dim_list = omics_dims[1]
        print(f"请打印出b_dim_list:{self.B_dim_list}")
        '''
        self.B_dim_list[i]是类FcSepVaeB实例的属性，它是一个包含23个整数值的yiwei列表，每个整数值对应于输入数据中染色体i+1的特征数。
        '''
        print("b_dim_list打印结束！")
        self.dim_1B = dim_1B

        # ENCODER
        # Layer 1
        self.encode_fc_1B_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1B_list.append(
                FCBlock(self.B_dim_list[i], dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B*23, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B*23, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 4
        self.decode_fc_4B_list = nn.ModuleList()
        for i in range(0, 23):
            self.decode_fc_4B_list.append(
                FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                        activation=False, normalization=False))

    def encode(self, x):
        level_2_B_list = []
        for i in range(0, 23):
            level_2_B_list.append(self.encode_fc_1B_list[i](x[1][i]))
        level_2_B = torch.cat(level_2_B_list, 1)

        level_3_B = self.encode_fc_2B(level_2_B)

        level_4 = self.encode_fc_3(level_3_B)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)

        level_3_B = self.decode_fc_3B(level_2)
        level_3_B_list = []
        for i in range(0, 23):
            level_3_B_list.append(level_3_B.narrow(1, self.dim_1B*i, self.dim_1B))

        recon_B_list = []
        for i in range(0, 23):
            recon_B_list.append(self.decode_fc_4B_list[i](level_3_B_list[i]))

        return [None, recon_B_list]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


# FcVae
class FcVaeABC(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input not separated by chromosome
    """
    def __init__(self, param, omics_dims, omics_subset_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=384, dim_2B=256,
                 dim_1A=384, dim_2A=256, dim_1C=384, dim_2C=256, dim_3=256, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeABC, self).__init__()

        if omics_subset_dims is not None:
            self.A_subset_dim = omics_subset_dims[0]
            self.B_subset_dim = omics_subset_dims[1]
            self.C_subset_dim = omics_subset_dims[2]

        # Decoder dimensions
        self.dim_1A = dim_1A // param.dec_reduction_factor ; self.dim_1B = dim_1B // param.dec_reduction_factor ; self.dim_1C = dim_1C // param.dec_reduction_factor
        self.dim_2A = dim_2A // param.dec_reduction_factor ; self.dim_2B = dim_2B // param.dec_reduction_factor ; self.dim_2C = dim_2C // param.dec_reduction_factor

        # Encoder dimensions
        dim_1A //= param.enc_reduction_factor ; dim_1B //= param.enc_reduction_factor ; dim_1C //= param.enc_reduction_factor
        dim_2B //= param.enc_reduction_factor ; dim_2B //= param.enc_reduction_factor ; dim_2C //= param.enc_reduction_factor
        
        self.A_dim = omics_dims[0]
        self.B_dim = omics_dims[1]
        self.C_dim = omics_dims[2]
        
        # ENCODER
        # Layer 1
        if omics_subset_dims is None:
            self.encode_fc_1B = FCBlock(self.B_dim, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                        activation=True)
            self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                        activation=True)
            self.encode_fc_1C = FCBlock(self.C_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                        activation=True)
        else:
            self.encode_fc_1B = FCBlock(self.B_subset_dim, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                        activation=True)
            self.encode_fc_1A = FCBlock(self.A_subset_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                        activation=True)
            self.encode_fc_1C = FCBlock(self.C_subset_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                        activation=True)
        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2C = FCBlock(dim_1C, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B+dim_2A+dim_2C, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, self.dim_2B+self.dim_2A+self.dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3B = FCBlock(self.dim_2B, self.dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3A = FCBlock(self.dim_2A, self.dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3C = FCBlock(self.dim_2C, self.dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4B = FCBlock(self.dim_1B, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_4A = FCBlock(self.dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_4C = FCBlock(self.dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B = self.encode_fc_1B(x[1])
        level_2_A = self.encode_fc_1A(x[0])
        level_2_C = self.encode_fc_1C(x[2])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3_C = self.encode_fc_2C(level_2_C)
        level_3 = torch.cat((level_3_B, level_3_A, level_3_C), 1)

        level_4 = self.encode_fc_3(level_3)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)
        level_2_B = level_2.narrow(1, 0, self.dim_2B)
        level_2_A = level_2.narrow(1, self.dim_2B, self.dim_2A)
        level_2_C = level_2.narrow(1, self.dim_2B+self.dim_2A, self.dim_2C)

        level_3_B = self.decode_fc_3B(level_2_B)
        level_3_A = self.decode_fc_3A(level_2_A)
        level_3_C = self.decode_fc_3C(level_2_C)

        recon_B = self.decode_fc_4B(level_3_B)
        recon_A = self.decode_fc_4A(level_3_A)
        recon_C = self.decode_fc_4C(level_3_C)

        return [recon_A, recon_B, recon_C]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcVaeAB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input not separated by chromosome

        这段代码定义了一个由全连接层组成的变分自编码器（VAE），其中包含编码器和解码器两个部分。编码器将输入数据映射到潜在空间中的均值和方差，然后使用 reparameterize
        函数对潜在向量进行采样，得到一个具有一定随机性和不确定性的潜在向量。解码器将潜在向量映射回原始数据空间，得到重构的数据。

        这个变分自编码器有两个输入，分别是两个不同的 omics 数据，输入数据经过编码器后合并，然后通过解码器进行重构。在编码器中，使用了 FCBlock 类来定义全连接层，
        其中包括了批归一化、Leaky ReLU 激活函数和 dropout 层等操作。在解码器中，也使用了 FCBlock 类来定义全连接层，但是不包括批归一化和 dropout 层。

        此外，这个变分自编码器还实现了 get_last_encode_layer 函数，用于获取编码器中的最后一层，也就是潜在向量的均值。在某些情况下，这个函数可以用于提取特征表示，
        供后续任务使用。
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=384, dim_2B=256,
                 dim_1A=384, dim_2A=256, dim_3=256, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeAB, self).__init__()

        self.A_dim = omics_dims[0]
        self.B_dim = omics_dims[1]
        self.dim_1B = dim_1B
        self.dim_2B = dim_2B
        self.dim_2A = dim_2A

        # ENCODER
        # Layer 1
        self.encode_fc_1B = FCBlock(self.B_dim, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B+dim_2A, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B+dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4B = FCBlock(dim_1B, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_4A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B = self.encode_fc_1B(x[1])
        level_2_A = self.encode_fc_1A(x[0])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3 = torch.cat((level_3_B, level_3_A), 1)

        level_4 = self.encode_fc_3(level_3)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)
        level_2_B = level_2.narrow(1, 0, self.dim_2B)
        level_2_A = level_2.narrow(1, self.dim_2B, self.dim_2A)

        level_3_B = self.decode_fc_3B(level_2_B)
        level_3_A = self.decode_fc_3A(level_2_A)

        recon_B = self.decode_fc_4B(level_3_B)
        recon_A = self.decode_fc_4A(level_3_A)

        return [recon_A, recon_B]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcVaeB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for DNA methylation dataset
        DNA methylation input not separated by chromosome
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=512, dim_2B=256,
                 dim_3=256, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeB, self).__init__()

        self.B_dim = omics_dims[1]

        # ENCODER
        # Layer 1
        self.encode_fc_1B = FCBlock(self.B_dim, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                      dropout_p=0, activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                         dropout_p=0, activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                   dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4B = FCBlock(dim_1B, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B = self.encode_fc_1B(x[1])

        level_3 = self.encode_fc_2B(level_2_B)

        level_4 = self.encode_fc_3(level_3)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)

        level_3_B = self.decode_fc_3B(level_2)

        recon_B = self.decode_fc_4B(level_3_B)

        return [None, recon_B]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcVaeA(nn.Module):
    """
        Defines a fully-connected variational autoencoder for gene expression dataset
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1A=1024, dim_2A=1024,
                 dim_3=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeA, self).__init__()

        self.A_dim = omics_dims[0]

        # ENCODER
        # Layer 1
        self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2A, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_A = self.encode_fc_1A(x[0]) # 这怎么x[0],多模态就是这样。

        level_3_A = self.encode_fc_2A(level_2_A)

        level_4 = self.encode_fc_3(level_3_A)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var) # 这是种高效的数值运算， 直接得到方差的开方
        eps = torch.randn_like(std) # eps 不是从 N（0，1）分布中取值的？ 这玩意就是从
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)

        level_3_A = self.decode_fc_3A(level_2)

        recon_A = self.decode_fc_4A(level_3_A)

        return [recon_A]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcVaeC(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        这是一个 PyTorch 模型类，定义了一个全连接变分自编码器（Fully-connected Variational Autoencoder）用于多组学数据集（multi-omics dataset）的降维和重构。
        它主要包含两个部分：编码器（Encoder）和解码器（Decoder）。编码器将输入数据映射到一个低维潜在空间（latent space），解码器则将潜在变量映射回重构数据。
    """
    def __init__(self, omics_dims, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1C=1024, dim_2C=1024, dim_3=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                omics_dims (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer, 标准化层，用于标准化输入数据
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
                dim_1C、dim_2C、dim_3 和 latent_dim：编码器和解码器中的隐藏层维度和潜在空间维度。
                dim_1C, dim_2C,dim_3 什么隐藏层？
        """

        super(FcVaeC, self).__init__()

        self.C_dim = omics_dims[2]  # omic_dims是输入数据的维度列表，omics_dims[2]是C_dim
        self.dim_2C = dim_2C

        # ENCODER
        # Layer 1
        self.encode_fc_1C = FCBlock(self.C_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2C = FCBlock(dim_1C, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2C, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        '''
        encode_fc_1C、encode_fc_2C、encode_fc_3：编码器中的全连接层，用于将输入数据映射到潜在空间
        '''
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)

        self.encode_fc_log_var = FCBlock(dim_3, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)
        '''
        encode_fc_mean、encode_fc_log_var：编码器中的全连接层，用于计算潜在空间的均值和标准差, 奇怪的是为什么这两个看上去一前一后，四线方法似乎完全一样，但是为什么输出却完全不同
        '''
        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3C = FCBlock(dim_2C, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 4
        self.decode_fc_4C = FCBlock(dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

        '''
        这里的三个全连接层是一种通用的设计模式，可以用来逐步抽象输入数据的特征。但是，并不是所有的情况下都需要三个全连接层，
        实际上，全连接层的数量和大小应该根据具体的任务和数据集进行调整。
        '''
    def encode(self, x):
        level_2_C = self.encode_fc_1C(x[2])

        level_3_C = self.encode_fc_2C(level_2_C)

        level_4 = self.encode_fc_3(level_3_C)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        '''
        在 reparameterize 函数中，eps 的作用是引入随机性，从而让生成的潜在向量具有一定的随机性。这是因为变分自编码器中的潜在向量是一种隐变量，
        其具有一定的不确定性，因此需要引入随机性来表示这种不确定性。具体地，eps 会与标准差 std 逐元素相乘，并加上均值 mean，从而生成一个新的潜在向量，
        其具有一定的随机性和不确定性。

        '''
        std = torch.exp(0.5 * log_var)
        print("参数重组中的std会是latent_dim大小的张量吗？")
        print(std)
        eps = torch.randn_like(std)
        print(eps)
        print("最复杂的数学计算结束！")
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)

        level_3_C = self.decode_fc_3C(level_2)

        recon_C = self.decode_fc_4C(level_3_C)

        return [None, None, recon_C]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        '''
        reparameterize 函数是用来对潜在向量进行重新参数化的函数，其中输入参数 mean 和 log_var 分别表示潜在向量的均值和对数方差。
        具体而言，reparameterize 函数通过从一个标准正态分布中采样，并且使用输入的均值和对数方差进行线性变换，从而生成一个新的潜在向量。
        '''
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


# Class for downstream task
class MultiFcClassifier(nn.Module):
    """
    Defines a multi-layer fully-connected classifier
    """
    def __init__(self, param, class_num=2, latent_dim=256, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0,
                 class_dim_1=128, class_dim_2=64, layer_num=3):
        """
        Construct a multi-layer fully-connected classifier
        Parameters:
            class_num (int)         -- the number of class
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
        """
        super(MultiFcClassifier, self).__init__()

        # down_reduction_factor 参数，这个参数是一个下采样因子，通常用于降低特征图的空间分辨率，从而减少网络的计算量和内存占用。
        class_dim_1 = class_dim_1 // param.down_reduction_factor
        class_dim_2 = class_dim_2 // param.down_reduction_factor
        # class_dim_1 和 class_dim_2 分别除以 down_reduction_factor 之后，表示中间层的维度缩小了 down_reduction_factor 倍。
        self.input_fc = FCBlock(latent_dim, class_dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                activation=True)
        # create a list to store fc blocks
        mul_fc_block = []
        # the block number of the multi-layer fully-connected block should be at least 3
        block_layer_num = max(layer_num, 3)
        input_dim = class_dim_1
        dropout_flag = True
        for num in range(0, block_layer_num-2):
            mul_fc_block += [FCBlock(input_dim, class_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout_p=dropout_flag*dropout_p, activation=True)]
            input_dim = class_dim_2
            # dropout for every other layer
            dropout_flag = not dropout_flag
        self.mul_fc = nn.Sequential(*mul_fc_block)

        # the output fully-connected layer of the classifier
        self.output_fc = FCBlock(class_dim_2, class_num, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                 activation=False, normalization=False)

    def forward(self, x):
        x1 = self.input_fc(x)
        x2 = self.mul_fc(x1)
        y = self.output_fc(x2)
        return y


class MultiFcRegression(nn.Module):
    """
    Defines a multi-layer fully-connected regression net
    """
    def __init__(self, latent_dim=256, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, down_dim_1=128,
                 down_dim_2=64, layer_num=3):
        """
        Construct a one dimensional multi-layer regression net
        Parameters:
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
        """
        super(MultiFcRegression, self).__init__()

        self.input_fc = FCBlock(latent_dim, down_dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                activation=True)

        # create a list to store fc blocks
        mul_fc_block = []
        # the block number of the multi-layer fully-connected block should be at least 3
        block_layer_num = max(layer_num, 3)
        input_dim = down_dim_1
        dropout_flag = True
        for num in range(0, block_layer_num-2):
            mul_fc_block += [FCBlock(input_dim, down_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope,
                             dropout_p=dropout_flag*dropout_p, activation=True)]
            input_dim = down_dim_2
            # dropout for every other layer
            dropout_flag = not dropout_flag
        self.mul_fc = nn.Sequential(*mul_fc_block)
        # the output fully-connected layer of the classifier
        self.output_fc = FCBlock(down_dim_2, 1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                 activation=False, normalization=False)
        '''
        在 MultiFcRegression 中，输出层 self.output_fc 的输出维度为 1，即该模型是一个回归模型，用于预测连续变量的取值。
        '''
    def forward(self, x):
        x1 = self.input_fc(x)
        x2 = self.mul_fc(x1)
        y = self.output_fc(x2)
        return y


class MultiFcSurvival(nn.Module):
    """
    Defines a multi-layer fully-connected survival predictor
    """
    def __init__(self, time_num=256, latent_dim=128, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0,
                 down_dim_1=512, down_dim_2=256, layer_num=3):
        """
        Construct a multi-layer fully-connected survival predictor
        Parameters:
            time_num (int)          -- the number of time intervals in the model
           “在 MultiFcSurvival 中，输出层 self.output_fc 的输出维度为 time_num，其中第一个维度表示生存时间，
            后面的 time_num - 1 个维度表示事件发生的概率。这些事件通常是死亡事件，因此这些概率也被称为死亡概率。” 这个不确定
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
        """
        super(MultiFcSurvival, self).__init__()
        # input_fc：输入层，将输入的特征向量通过一个全连接（FC）层进行变换，并使用批归一化（Batch Normalization）和激活函数（Tanh）进行规范化和非线性变换。
        self.input_fc = FCBlock(latent_dim, down_dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                activation=True, activation_name='Tanh')

        # create a list to store fc blocks
        mul_fc_block = [] # 这是个包含多个全连接块的列表，每个全连接块都是使用 FCBlock 类进行构建的
        # mul_fc：多层全连接块，将输入的特征向量通过若干个全连接层进行变换，每个全连接层后都使用批归一化和激活函数进行规范化和非线性变换，
        # 这些全连接层的输出都作为下一个全连接层的输入。
        # the block number of the multi-layer fully-connected block should be at least 3
        block_layer_num = max(layer_num, 3)  # block_layer_num表示要构造的全连接块的层数， layer_num是模型定义时传入的参数，表示原本要构造的全连接块的层数，但是如果传入的layer_num小于3，那么为了保证模型的有效性，就会将block_layer_num设为3
        input_dim = down_dim_1 # input_dim 当前要构造的全连接层的输入维度，初始值为down_dim_1
        dropout_flag = True # dropout_flag表示当前是否要应用dropout，初始值为True，因为第一个全连接层需要应用dropout。
        for num in range(0, block_layer_num - 2):
            mul_fc_block += [FCBlock(input_dim, down_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout_p=dropout_p, activation=True, activation_name='Tanh')]  # 每个multi fc block都使用FCBlock类进行构建，其中包括批归一化、激活函数和dropout。
            input_dim = down_dim_2
            # dropout for every other layer
            dropout_flag = not dropout_flag
            '''
            在循环中，从第1个全连接层开始，到第block_layer_num-2个全连接层结束，每个全连接层的输入维度都是上一个全连接层的输出维度input_dim，输出维度都是down_dim_2
            '''
        self.mul_fc = nn.Sequential(*mul_fc_block)
        '''
        1. * 运算符有多种用途，其中之一是展开可迭代对象。当使用 * 运算符对列表进行操作时，它会将列表中的元素逐个取出来，然后作为单独的参数传递给函数或构造函数。
        2. nn.Sequential 类可以接受一个包含多个网络层或模块的列表作为输入，并按照列表中的顺序将这些网络层或模块连接起来。因此，在这个代码段中，mul_fc_block 
        中的多个全连接块可以通过 * 运算符展开为一个列表，然后作为参数传递给 nn.Sequential 类的构造函数中，最终形成一个序列化的神经网络模块 self.mul_fc。
        '''
        # the output fully-connected layer of the classifier
        # the output dimension should be the number of time intervals
        self.output_fc = FCBlock(down_dim_2, time_num, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                 activation=False, normalization=False)
        # 定义了 self.output_fc 的全连接块，用于将输入张量的维度从 down_dim_2 转换为 time_num。

    def forward(self, x):
        x1 = self.input_fc(x)
        x2 = self.mul_fc(x1)
        y = self.output_fc(x2)
        return y
    '''
     PyTorch 中神经网络模型必须实现的一个方法，用于定义前向传播的计算过程。在这个代码段中，forward 函数定义了整个模型的前向传播过程，
     即将输入张量 x 从输入层传递到输出层，并返回最终的输出结果。
    '''

class MultiFcMultitask(nn.Module):
    """
        Defines a multi-layer fully-connected multitask downstream network
    """

    def __init__(self, class_num=2, time_num=256, latent_dim=128, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0,
                 layer_num=3):
        """
        Construct a multi-layer fully-connected multitask downstream network
        Parameters:
            class_num (int)         -- the number of class
            time_num (int)          -- the number of time intervals in the model
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the downstream networks, >=3
        """
        super(MultiFcMultitask, self).__init__()
        norm_layer_none = lambda x: Identity()
        self.survival = MultiFcSurvival(time_num, latent_dim, norm_layer=norm_layer_none, leaky_slope=leaky_slope, dropout_p=0.5, layer_num=layer_num)
        self.classifier = MultiFcClassifier(class_num, latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=leaky_slope, dropout_p=0.2, layer_num=layer_num)
        self.regression = MultiFcRegression(latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=leaky_slope, dropout_p=0.01, layer_num=layer_num)

    def forward(self, x):
        y_out_sur = self.survival(x)
        y_out_cla = self.classifier(x)
        y_out_reg = self.regression(x)
        return y_out_sur, y_out_cla, y_out_reg


class MultiFcAlltask(nn.Module):
    """
    这段代码定义了一个多层全连接神经网络，用于多任务学习。该神经网络包含三个部分：生存分析模块、多分类模块和回归模块。
    生存分析模块由MultiFcSurvival类实现，
    多分类模块由MultiFcClassifier类实现，
    回归模块由MultiFcRegression类实现。
    """

    def __init__(self, class_num, time_num=256, task_num=7, latent_dim=128, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0,
                 layer_num=3):
        """
        Construct a multi-layer fully-connected multitask downstream network (all tasks)
        Parameters:

            class_num (list)        -- the list of class numbers
            time_num (int)          -- the number of time intervals in the model
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
            task_num (int)          -- the number of downstream tasks
        """
        super(MultiFcAlltask, self).__init__()
        norm_layer_none = lambda x: Identity()
        '''
        在初始化函数中，根据输入的参数设置了网络的各个层，包括生存分析模块、多分类模块和回归模块，
        其中生存分析模块和回归模块的输出维度为latent_dim，
        多分类模块的输出维度为各个任务的类别数（由class_num列表给出）。norm_layer参数表示网络中的标准化层类型，
        leaky_slope表示Leaky ReLU激活函数的负斜率，dropout_p表示dropout层的概率，layer_num表示分类器的层数，task_num表示下游任务的数量。
        '''
        self.survival = MultiFcSurvival(time_num, latent_dim, norm_layer=norm_layer_none, leaky_slope=leaky_slope, dropout_p=0.5, layer_num=layer_num)
        self.classifiers = nn.ModuleList([MultiFcClassifier(class_num[i], latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=leaky_slope, dropout_p=0.2, layer_num=layer_num) for i in range(task_num-2)])
        self.regression = MultiFcRegression(latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=leaky_slope, dropout_p=0.01, layer_num=layer_num)
        self.task_num = task_num
        '''
        surival = MultiFcSurvival(time_num, latent_dim, norm_layer)
        time_num：表示时间区间的数量，是输入数据的一个维度。
        latent_dim：表示潜在空间的维度，也是输出数据的维度。
        norm_layer：表示网络中的标准化层类型，默认为None。
        leaky_slope：表示Leaky ReLU激活函数的负斜率，默认为0.2。
        dropout_p：表示dropout层的概率，默认为0.5。
        layer_num：表示分类器的层数，默认为3。
        
        classifiers = nn.ModuleList([MultiFcClassifier(class_num[i],latent_dim, latent_dim, norm_layer==nn.BatchNorm1d,leaky_slope=leaky_slope,dropout_p=0.2, layer_num=layer_num) for i in range(task_num-2)] 
        迭代对象是range(task_num-2)，即从0到task_num-3的整数序列。
        这表示对于每个下游任务，都创建一个MultiFcClassifier类的实例。
        class_num[i]表示第i个下游任务的类别数，latent_dim表示输入数据和输出数据的维度，
        norm_layer表示网络中的标准化层类型，leaky_slope表示Leaky ReLU激活函数的负斜率，dropout_p表示dropout层的概率，layer_num表示分类器的层数。
        
        regression = MultiFcRegression(latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=leaky_slope,layer_num=layer_num)
        MultiFcRegression类是一个自定义的PyTorch模型类，用于回归任务，其参数说明如下：
        latent_dim：输入数据和输出数据的维度。
        norm_layer：标准化层的类型，这里使用的是nn.BatchNorm1d类，表示在每个mini-batch中对每个特征进行标准化。
        leaky_slope：Leaky ReLU激活函数的负斜率，一般设为一个小的正数，如0.2。
        dropout_p：dropout层的概率，表示在训练过程中随机丢弃一定比例的神经元，以减轻过拟合的影响。
        layer_num：网络的层数，表示网络中包含几个全连接层。
        '''

    def forward(self, x):
        y_out_sur = self.survival(x)
        y_out_cla = []
        for i in range(self.task_num - 2):
            y_out_cla.append(self.classifiers[i](x))
        y_out_reg = self.regression(x)
        return y_out_sur, y_out_cla, y_out_reg


# Class for the OmiEmbed combined network
class OmiEmbed(nn.Module):
    """
    Defines the OmiEmbed combined network
    该模型结合了变分自编码器（VAE）和下游任务网络（downstream task network），用于多组学数据的嵌入和下游任务预测
    """
    def __init__(self, net_VAE, net_down, omics_dims, omics_mode='multi_omics', norm_layer=nn.InstanceNorm1d, filter_num=8, kernel_size=9,
                 leaky_slope=0.2, dropout_p=0, latent_dim=128, class_num=2, time_num=256, task_num=7):
        """
            Construct the OmiEmbed combined network
            Parameters:
                net_VAE (str)           -- the backbone of the VAE, default: conv_1d
                net_down (str)          -- the backbone of the downstream task network, default: multi_FC_classifier
                omics_dims (list)       -- the list of input omics dimensions
                omics_mode (str)        -- omics types would like to use in the model
                norm_layer              -- normalization layer
                filter_num (int)        -- the number of filters in the first convolution layer in the VAE
                kernel_size (int)       -- the kernel size of convolution layers
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
                class_num (int/list)    -- the number of classes,输出的class 好像改变了
                time_num (int)          -- the number of time intervals
                task_num (int)          -- the number of downstream tasks
        """
        super(OmiEmbed, self).__init__()

        self.vae = None # VAE的类型
        if net_VAE == 'conv_1d':
            if omics_mode == 'abc':
                self.vae = ConvVaeABC(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'ab':
                self.vae = ConvVaeAB(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'b':
                self.vae = ConvVaeB(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'a':
                self.vae = ConvVaeA(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'c':
                self.vae = ConvVaeC(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p, latent_dim=latent_dim)

        elif net_VAE == 'fc_sep':
            '''
            fc_sep : 
            1.更好地处理变长输入序列：在自然语言处理任务中，输入序列的长度通常是可变的，这使得传统的全连接层实现方式难以处理。
            而将全连接层的乘法操作分解成两个矩阵乘法可以更好地处理变长输入序列，因为它将输入张量分解成两个张量，其中一个张量与一个固定的权重矩阵相乘，
            另一个张量与另一个可学习的权重矩阵相乘，从而可以灵活地处理输入序列的长度。
    
            2.减少模型参数：将全连接层的乘法操作分解成两个矩阵乘法可以减少模型参数，因为它可以使用两个较小的权重矩阵来代替一个较大的权重矩阵。
            这可以减少模型的存储需求和计算成本，从而提高模型的效率。

            3.提高模型的表达能力：将全连接层的乘法操作分解成两个矩阵乘法可以提高模型的表达能力，因为它允许模型学习不同的权重矩阵来捕捉输入张量中的不同特征。
            这可以使模型更好地适应不同的输入数据，并提高模型的泛化能力。
            '''
            if omics_mode == 'abc':
                self.vae = FcSepVaeABC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'ab':
                self.vae = FcSepVaeAB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'b':
                self.vae = FcSepVaeB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'a':
                self.vae = FcVaeA(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'c':
                self.vae = FcVaeC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)

        elif net_VAE == 'fc':
            if omics_mode == 'abc':
                self.vae = FcVaeABC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'ab':
                self.vae = FcVaeAB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'b':
                self.vae = FcVaeB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'a':
                self.vae = FcVaeA(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
            elif omics_mode == 'c':
                self.vae = FcVaeC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        else:
            raise NotImplementedError('VAE model name [%s] is not recognized' % net_VAE)

        self.net_down = net_down # 下游网络的类型？
        self.down = None
        if net_down == 'multi_FC_classifier':
            self.down = MultiFcClassifier(class_num, latent_dim, norm_layer, leaky_slope, dropout_p)
        elif net_down == 'multi_FC_regression':
            self.down = MultiFcRegression(latent_dim, norm_layer, leaky_slope, dropout_p)
        elif net_down == 'multi_FC_survival':
            self.down = MultiFcSurvival(time_num, latent_dim, norm_layer, leaky_slope, dropout_p)
        elif net_down == 'multi_FC_multitask':
            self.down = MultiFcMultitask(class_num, time_num, latent_dim, norm_layer, leaky_slope, dropout_p)
        elif net_down == 'multi_FC_alltask':
            self.down = MultiFcAlltask(class_num, time_num, task_num, latent_dim, norm_layer, leaky_slope, dropout_p)
        else:
            raise NotImplementedError('Downstream model name [%s] is not recognized' % net_down)

    def get_last_encode_layer(self):
        return self.vae.get_last_encode_layer()
    '''
    这段代码看起来是一个类方法，其中的self.vae是一个变量，表示类中的一个属性或者另一个对象。
    根据代码中的函数名和参数，可以猜测这个方法的作用是获取某个模型的最后一个编码层。
    '''

    def forward(self, x):
        '''
        模型的输入是多组学数据x，输出是嵌入向量z，重构后的数据recon_x，以及VAE的输出结果mean和log_var。
        如果选择了多任务或全任务下游网络，则还会输出生存预测y_out_sur，分类预测y_out_cla和回归预测y_out_reg。
        '''
        z, recon_x, mean, log_var = self.vae(x)
        if self.net_down == 'multi_FC_multitask' or self.net_down == 'multi_FC_alltask':
            y_out_sur, y_out_cla, y_out_reg = self.down(mean)
            return z, recon_x, mean, log_var, y_out_sur, y_out_cla, y_out_reg
        else:
            y_out = self.down(mean)
            return z, recon_x, mean, log_var, y_out  # recon_x 是重构过后的数据


def get_norm_layer(norm_type='batch'):
    """
    这是一个Python函数，名为get_norm_layer，用于返回一个归一化层。该函数的参数norm_type指定了所需的归一化类型，包括batch、instance和none。
    如果norm_type为batch，则返回一个nn.BatchNorm1d层，如果为instance，则返回一个nn.InstanceNorm1d层，如果为none，则返回一个恒等函数Identity()，该函数不进行任何归一化操作。
    Return a normalization layer
    Parameters:
        norm_type (str) -- the type of normalization applied to the model, default to use batch normalization, options: [batch | instance | none ]
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization method [%s] is not found' % norm_type)
    return norm_layer


def define_net(net_VAE, net_down, omics_dims, omics_mode='multi_omics', norm_type='batch', filter_num=8, kernel_size=9,
               leaky_slope=0.2, dropout_p=0, latent_dim=256, class_num=2, time_num=256, task_num=7, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Create the OmiEmbed network
    这是一个Python函数，名为define_net，用于创建一个名为OmiEmbed的神经网络模型，该模型包括一个变分自编码器（VAE）和一个下游任务网络，
    用于对多组学数据进行嵌入学习（embeddings）和下游任务分类。具体来说，该函数的参数包括：

    Parameters:
        net_VAE：VAE的主干网络，默认为conv_1d；
        net_down：下游任务网络的主干网络，默认为multi_FC_classifier；
        net_VAE (str)           -- the backbone of the VAE, default: conv_1d
        net_down (str)          -- the backbone of the downstream task network, default: multi_FC_classifier
        omics_dims (list)       -- the list of input omics dimensions
        omics_mode (str)        -- omics types would like to use in the model
        norm_type (str)         -- the name of normalization layers used in the network, default: batch
        filter_num (int)        -- the number of filters in the first convolution layer in the VAE
        kernel_size (int)       -- the kernel size of convolution layers
        leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
        latent_dim (int)        -- the dimensionality of the latent space
        class_num (int)         -- the number of classes
        time_num (int)          -- the number of time intervals
        task_num (int)          -- the number of downstream tasks
        init_type (str)         -- the name of our initialization method
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal initialization methods
        gpu_ids (int list)      -- which GPUs the network runs on: e.g., 0,1

    Returns the OmiEmbed network

    The network has been initialized by <init_net>.
    """

    net = None

    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)

    net = OmiEmbed(net_VAE, net_down, omics_dims, omics_mode, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p,
                   latent_dim, class_num, time_num, task_num)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_VAE(param, net_VAE, omics_subset_dims, omics_dims, omics_mode='multi_omics', norm_type='batch', filter_num=8, kernel_size=9, leaky_slope=0.2, dropout_p=0,
               latent_dim=256, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Create the VAE network
    这是一个用于创建变分自编码器（Variational Autoencoder，VAE）的Python函数。VAE是一种生成模型，它可以学习输入数据的潜在分布，
    并生成新的与原始数据相似的样本。该函数的参数包括VAE的结构参数、输入数据的维度、使用的正则化方式、网络初始化方式等。该函数使用PyTorch框架实现，
    并返回一个初始化后的VAE网络。该函数中的不同VAE模型（如conv_1d、fc_sep、fc）是不同的VAE结构，可根据具体需求进行选择。
    conv_1d,fc_sep,fc 都是什么呀？
    Parameters:
        net_VAE (str)           -- the backbone of the VAE, default: conv_1d
        omics_dims (list)       -- the list of input omics dimensions
        omics_mode (str)        -- omics types would like to use in the model
        norm_type (str)         -- the name of normalization layers used in the network, default: batch
        filter_num (int)        -- the number of filters in the first convolution layer in the VAE
        kernel_size (int)       -- the kernel size of convolution layers
        leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
        latent_dim (int)        -- the dimensionality of the latent space
        init_type (str)         -- the name of our initialization method
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal initialization methods
        gpu_ids (int list)      -- which GPUs the network runs on: e.g., 0,1

    Returns a VAE

    The default backbone of the VAE is one dimensional convolutional layer.

    The generator has been initialized by <init_net>.
    """

    net = None

    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)

    if net_VAE == 'conv_1d':
        if omics_mode == 'abc':
            net = ConvVaeABC(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p,
                             latent_dim=latent_dim)
        elif omics_mode == 'ab':
            net = ConvVaeAB(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p,
                            latent_dim=latent_dim)
        elif omics_mode == 'b':
            net = ConvVaeB(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p,
                           latent_dim=latent_dim)
        elif omics_mode == 'a':
            net = ConvVaeA(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p,
                           latent_dim=latent_dim)
        elif omics_mode == 'c':
            net = ConvVaeC(omics_dims, norm_layer, filter_num, kernel_size, leaky_slope, dropout_p,
                           latent_dim=latent_dim)

    elif net_VAE == 'fc_sep':
        if omics_mode == 'abc':
            net = FcSepVaeABC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'ab':
            net = FcSepVaeAB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'b':
            net = FcSepVaeB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'a':
            net = FcVaeA(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'c':
            net = FcVaeC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)

    elif net_VAE == 'fc':
        if omics_mode == 'abc':
            net = FcVaeABC(param, omics_dims, omics_subset_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'ab':
            net = FcVaeAB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'b':
            net = FcVaeB(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'a':
            net = FcVaeA(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
        elif omics_mode == 'c':
            net = FcVaeC(omics_dims, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
    else:
        raise NotImplementedError('VAE model name [%s] is not recognized' % net_VAE)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_down(param, net_down, norm_type='batch', leaky_slope=0.2, dropout_p=0, latent_dim=256, class_num=2, time_num=256,
                task_num=7, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
        Create the downstream task network

        Parameters:
            net_down (str)          -- the backbone of the downstream task network, default: multi_FC_classifier
            norm_type (str)         -- the name of normalization layers used in the network, default: batch
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            class_num (int)         -- the number of class
            time_num (int)          -- the number of time intervals
            task_num (int)          -- the number of downstream tasks
            init_type (str)         -- the name of our initialization method
            init_gain (float)       -- scaling factor for normal, xavier and orthogonal initialization methods
            gpu_ids (int list)      -- which GPUs the network runs on: e.g., 0,1

        Returns a downstream task network

        The default downstream task network is a multi-layer fully-connected classifier.

        The generator has been initialized by <init_net>.
        """

    net = None

    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)

    if net_down == 'multi_FC_classifier':
        net = MultiFcClassifier(param, class_num, latent_dim, norm_layer, leaky_slope, dropout_p)
    elif net_down == 'multi_FC_regression':
        net = MultiFcRegression(latent_dim, norm_layer, leaky_slope, dropout_p)
    elif net_down == 'multi_FC_survival':
        net = MultiFcSurvival(time_num, latent_dim, norm_layer, leaky_slope, dropout_p)
    elif net_down == 'multi_FC_multitask':
        net = MultiFcMultitask(class_num, time_num, latent_dim, norm_layer, leaky_slope, dropout_p)
    elif net_down == 'multi_FC_alltask':
        net = MultiFcAlltask(class_num, time_num, task_num, latent_dim, norm_layer, leaky_slope, dropout_p)
    else:
        raise NotImplementedError('Downstream model name [%s] is not recognized' % net_down)

    return init_net(net, init_type, init_gain, gpu_ids)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Initialize a network:
    1. register CPU/GPU device (with multi-GPU support);
    2. initialize the network weights
    Parameters:
        net (nn.Module)    -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # multi-GPUs
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (nn.Module)    -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier_normal | xavier_uniform | kaiming_normal | kaiming_uniform | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
    """
    # define the initialization function
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('Initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_scheduler(optimizer, param):
    """
    Return a learning rate scheduler

    Parameters:
        optimizer (opt class)     -- the optimizer of the network
        param (params class)      -- param.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <param.niter> epochs and linearly decay the rate to zero
    over the next <param.niter_decay> epochs.
    这段代码是一个函数get_scheduler，
    用于根据指定的学习率策略（linear、step、plateau、cosine）返回相应的学习率调度器（scheduler）
    """
    if param.lr_policy == 'linear':
        def lambda_rule(epoch):
            '''
            epoch : 当前的epoch
            epoch_count : 训练开始之前的epoch数量
            epoch_num : 总训练轮数
            epoch_num_decay: 学利率衰减的论数
            '''
            lr_lambda = 1.0 - max(0, epoch + param.epoch_count - param.epoch_num + param.epoch_num_decay) / float(param.epoch_num_decay + 1)
            return lr_lambda
        # lr_scheduler is imported from torch.optim
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) # 算线性学习率调度器的衰减因子 lr_lambda 是什么意思？
        '''
        当学习率策略是linear时，将使用线性衰减的学习率调度器。该调度器在前param.niter个epoch保持学习率不变，
        并在接下来的param.niter_decay个epoch中线性地将学习率衰减为0。函数中定义了一个lambda_rule函数，
        用于计算每个epoch的学习率衰减因子，并将其作为参数传递给lr_scheduler.LambdaLR函数，从而创建一个LambdaLR类型的学习率调度器。
        '''
    elif param.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=param.decay_step_size, gamma=0.1)
        # 当学习率策略是step时，将使用StepLR类型的学习率调度器。该调度器将在每个param.decay_step_size个epoch后将学习率乘以0.1。
    elif param.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        # 当学习率策略是plateau时，将使用ReduceLROnPlateau类型的学习率调度器。
        # 该调度器将监测给定度量（mode='min'时为验证损失，mode='max'时为验证准确率）的结果，如果连续patience个epoch未能看到结果改善，则将学习率乘以factor。
        # 为什么这么复杂？
    elif param.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=param.epoch_num, eta_min=0)
        # 当学习率策略是cosine时，将使用CosineAnnealingLR类型的学习率调度器。该调度器将在T_max个epoch后将学习率降为0，采用余弦退火策略进行学习率调整。
    else:
        return NotImplementedError('Learning rate policy [%s] is not found', param.lr_policy)
    # 如果输入的学习率策略不在上述四种中，则将返回一个NotImplementedError异常，提示该学习率策略未实现。
    return scheduler
    # 总之，该函数的作用是返回一个相应的学习率调度器，用于在训练过程中动态调整学习率，从而提高模型的性能。
