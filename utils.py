import time
import torch
from tqdm import tqdm


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dict = {}
    with open(file_path, 'r', encoding='UTF-8') as fp:
        for line in tqdm(fp):
            lin = line.strip()
            if not lin:
                continue