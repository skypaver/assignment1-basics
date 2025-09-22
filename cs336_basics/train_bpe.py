import regex as re
from typing import Dict, Tuple, List
from collections import Counter
import logging
from tqdm import tqdm

import cs336_basics.utils.io as io
import cs336_basics.utils.tokenizer_utils as t_utils
import cs336_basics.bpe_tokenizer as bpe

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',  # 可以包含时间戳
        handlers=[
            logging.FileHandler("../logs/train_bpe_logs.txt"),  # 写入文件
            # logging.StreamHandler()  # 同时输出到控制台
        ]
    )


if __name__ == "__main__":
    filepath = "../data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = [
        '<|endoftext|>',
        # '<|fim_prefix|>',
        # '<|fim_middle|>',
        # '<|fim_suffix|>',
        # '<|endofprompt|>'
    ]

    tokenizer = bpe.BPETokenizer()
    tokenizer.train(filepath, 20000, 8, special_tokens, True)





