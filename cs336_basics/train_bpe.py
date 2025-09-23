from typing import Dict, Tuple
import logging

import cs336_basics.bpe_tokenizer as bpe

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',  # 可以包含时间戳
        handlers=[
            logging.FileHandler("../logs/train_bpe_logs.txt"),  # 写入文件
            # logging.StreamHandler()  # 同时输出到控制台
        ]
    )


def evaluate_tokenizer(vocab: Dict[int, bytes], merges: Dict[Tuple[bytes, bytes], int]):
    unique_tokens = set(vocab.values())
    print(f"词汇表大小: {len(vocab):,}")
    print(f"唯一token数: {len(unique_tokens):,}")
    print(f"合并操作数: {len(merges):,}")


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
    tokenizer.train(filepath, 20000, 8, special_tokens, progress_bar=True)

    evaluate_tokenizer(tokenizer.vocab, tokenizer.merges)





