import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import copy


# 深拷贝
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed, tgt_embed, generator):
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder  # 编码器模块
        self.decoder = decoder  # 解码器模块
        self.src_embed = src_embed  # 源语言嵌入层（将源词转为向量）
        self.tgt_embed = tgt_embed  # 目标语言嵌入层（将目标词转为向量）
        self.generator = generator  # 生成器模块（输出词表概率）

    def forward(self,src, tgt, src_mask, tgt_mask):

        self.tmp = self.encoder(src,src_mask)
        return self.decoder(self.tmp, src_mask, tgt, tgt_mask)

    def encoder(self,src,src_mask):
        """
            对源序列进行编码，生成源序列的语义记忆（memory）
            核心逻辑：源序列token → 词嵌入向量 → 编码器编码（结合掩码）

            Args:
                src (torch.Tensor): 源序列token张量，形状一般为 [batch_size, src_seq_len]
                src_mask (torch.Tensor): 源序列掩码张量，用于遮挡padding部分，形状 [batch_size, 1, src_seq_len]

            Returns:
                torch.Tensor: 编码器输出的语义记忆（memory），形状 [batch_size, src_seq_len, d_model]
                              包含源序列的全部语义信息，供解码器使用
        """
        return self.encoder(self.src_embed(src),src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
            基于编码器的语义记忆对目标序列进行解码，生成目标序列的特征表示
            核心逻辑：目标序列token → 词嵌入向量 → 解码器解码（结合memory和各类掩码）

            Args:
                memory (torch.Tensor): 编码器输出的源序列语义记忆，形状 [batch_size, src_seq_len, d_model]
                src_mask (torch.Tensor): 源序列掩码张量，用于解码器的跨注意力层，避免关注源序列padding
                tgt (torch.Tensor): 目标序列token张量，形状一般为 [batch_size, tgt_seq_len]
                tgt_mask (torch.Tensor): 目标序列掩码张量，遮挡padding和未来位置，形状 [batch_size, tgt_seq_len, tgt_seq_len]

            Returns:
                torch.Tensor: 解码器输出的目标序列特征，形状 [batch_size, tgt_seq_len, d_model]
                              需传入Generator层才能生成词表概率
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        # d_model：解码器输出特征的维度（论文中为512）
        # vocab：目标词表的大小（比如翻译任务中目标语言的总词数）
        super(Generator,self).__init__()
        # 定义线性投影层：将d_model维特征映射到词表维度
        self.proj = nn.Linear(d_model, vocab)

    def forward(self,x):
        # 1. self.proj(x)：线性层将特征从d_model维 → vocab维
        # 2. log_softmax(..., dim=-1)：对最后一维做log_softmax，将数值转为对数概率
        return log_softmax(self.proj(x), dim=-1)

# 归一化模块
class LayerNorm(nn.Module):
    def __init__(self,features, eps=1e-6):
        super(LayerNorm.self).__init__()
        # 可学习的缩放参数（a_2）：初始化为全1，形状 [features]
        self.a_2 = nn.Parameter(torch.ones(features))
        # 可学习的偏移参数（b_2）：初始化为全0，形状 [features]
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 极小值eps：防止分母为0（除以std时），默认1e-6
        self.eps = eps

    def forward(self,x):
        # 对张量的最后一个维度
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Encoder(nn.Module):
    """
        Transformer完整编码器：由N个相同的EncoderLayer堆叠而成
        核心逻辑：源序列特征 → 逐层经过N个EncoderLayer → 层归一化 → 最终语义表征
    """
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)


    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)