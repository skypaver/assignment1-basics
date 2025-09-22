from typing import Dict, Tuple, List
from collections import Counter
import regex as re
import concurrent.futures
from tqdm import tqdm

import cs336_basics.utils.io as io


def _pre_tokenize(doc: str) -> Dict[str, int]:
    compiled_pattern = re.compile(io.GPT2_PRETOKENIZER_PATTERN)
    return Counter(re.findall(compiled_pattern, doc))


def pre_tokenize(filepath: str, num_processes: int, special_tokens: List[str]) -> Dict[Tuple[bytes, ...], int]:
    text = io.load_and_sample_file(filepath)

    for token in special_tokens:
        text = text.replace(token, "")

    if num_processes == 1:
        pre_tokens = _pre_tokenize(text)

    else:
        chunk_size = len(text) // num_processes
        texts = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            pre_tokens = executor.map(_pre_tokenize, texts)

        pre_tokens = sum(pre_tokens, Counter())

    pre_token_freq = {}
    for pre_token, freq in pre_tokens.items():
        pre_token_freq[tuple([bytes([b]) for b in pre_token.encode("utf-8")])] = freq

    return pre_token_freq


def evaluate_tokenizer(vocab: Dict[int, bytes], merges: Dict[Tuple[bytes, bytes], int]):
    unique_tokens = set(vocab.values())
    print(f"词汇表大小: {len(vocab):,}")
    print(f"唯一token数: {len(unique_tokens):,}")
    print(f"合并操作数: {len(merges):,}")
