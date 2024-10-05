import math
import random
import torch
import numpy as np
from inspect import isfunction


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def default(val, d):
    def exists(x):
        return x is not None
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cosine_similarity(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    norm_vector1 = math.sqrt(sum(a * a for a in vector1))
    norm_vector2 = math.sqrt(sum(b * b for b in vector2))
    return dot_product / (norm_vector1 * norm_vector2)


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 固定cuda的随机数种子，每次返回的卷积算法将是确定的
    torch.backends.cudnn.deterministic = True


def random_generate_sequence(len_list=None, n=100, seed=None, min_length=5, max_length=50):
    random.seed(seed)
    """
    随机生成序列
    """
    amino_list = list("MEKVCSDQGNWTIPLYRHFA")
    seq_list = []
    if len_list is None:
        for i in range(n):
            # 确定序列的长度
            seq_len = random.randint(min_length, max_length)
            seq = ""
            for j in range(seq_len):
                seq = seq.join(amino_list[random.randint(0, 19)])
            seq_list.append(seq)
    else:
        for seq_len in len_list:
            seq = ""
            for j in range(seq_len):
                seq = seq.join(amino_list[random.randint(0, 19)])
            seq_list.append(seq)
    return seq_list
