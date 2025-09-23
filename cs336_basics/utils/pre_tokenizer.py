from typing import Dict, Tuple, List
from collections import Counter
import regex as re
import concurrent.futures

import cs336_basics.utils.io as io


def _pre_tokenize(doc: str) -> Dict[str, int]:
    compiled_pattern = re.compile(io.GPT2_PRETOKENIZER_PATTERN)
    return Counter(re.findall(compiled_pattern, doc))


def pre_tokenize(filepath: str, num_processes: int, special_tokens: List[str], sample_size: int) -> Dict[Tuple[bytes, ...], int]:
    text = io.load_and_sample_file(filepath, sample_size=sample_size)

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
