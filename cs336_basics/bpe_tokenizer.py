import heapq
from collections import Counter
from typing import Dict, Tuple, List
import pickle
import regex as re
from tqdm import tqdm
import logging

import cs336_basics.utils.io as io
import cs336_basics.utils.tokenizer_utils as t_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',  # 可以包含时间戳
    handlers=[
        logging.FileHandler("../logs/train_bpe_logs.txt"),  # 写入文件
        # logging.StreamHandler()  # 同时输出到控制台
    ]
)


class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}

        self.special_tokens = {}  # str -> int
        self.inverse_special_tokens = {}

        self.pattern = io.GPT2_PRETOKENIZER_PATTERN
        self.compiled_pattern = re.compile(self.pattern)

        self.heap = []

    def train(self, filepath: str, vocab_size: int, num_processes: int, special_tokens: List, progress_bar=False):
        logging.info(f"<--- New Train --->")
        vocab = {i: bytes([i]) for i in range(256)}
        for i, token in enumerate(special_tokens):
            vocab[256 + i] = token.encode("utf-8")
            self.special_tokens[vocab[256 + i]] = 256 + i
            self.inverse_special_tokens[256 + i] = vocab[256 + i]

        pre_token_freq = t_utils.pre_tokenize(filepath, num_processes, special_tokens)
        # print(pre_token_freq)

        pair_freq = self.get_token_pair_freq(pre_token_freq, progress_bar)
        logging.info(f"size of pair_freq: {len(pair_freq)}")
        # print(pair_freq)

        pbar = tqdm(total=vocab_size - len(vocab)) if progress_bar else None
        merges = {}
        while len(vocab) < vocab_size:
            most_freq_pair = self.get_most_freq_pair(pair_freq)
            if most_freq_pair is None:
                break

            # update vocab
            idx = len(vocab)
            vocab[idx] = b"".join(most_freq_pair)
            # merge & update pair_freq
            merges[b"".join(most_freq_pair)] = idx
            pre_token_freq = self.merge_and_update_freq(pre_token_freq, most_freq_pair, pair_freq)
            pbar.update(len(vocab) - 256 - len(self.special_tokens) - pbar.n) if progress_bar else None
        pbar.close() if progress_bar else None

        self.vocab = vocab
        self.merges = merges

        self.save()

        return

    def _encode_chunk(self, text: str):
        if text in self.special_tokens:
            return [self.special_tokens[text]]
        else:
            text_chunks = re.findall(self.compiled_pattern, text)
            result = []
            for chunk in text_chunks:
                chunk_bytes = chunk.encode("utf-8")
                chunk_ids = list(chunk_bytes)
                # print(f"chunk_bytes: {chunk_bytes}, chunk_ids: {chunk_ids}")

                while len(chunk_ids) > 1:
                    pairs = set()
                    for p in zip(chunk_ids[:-1], chunk_ids[1:]):
                        pairs.add(p)
                    merge_pair = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
                    if merge_pair not in self.merges:
                        break

                    new_id = self.merges[merge_pair]
                    new_ids = []
                    i = 0
                    while i < len(chunk_ids):
                        curr_pair = tuple(chunk_ids[i:i + 2])
                        if curr_pair == merge_pair:
                            new_ids.append(new_id)
                            i += 1
                        else:
                            new_ids.append(chunk_ids[i])
                        i += 1
                    chunk_ids = new_ids
                result.extend(chunk_ids)
            return result

    def encode(self, text: str):
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            special_split_chunks = re.split(special_pattern, text)
        else:
            special_split_chunks = [text]

        print(f"special_split_chunks: {special_split_chunks}")

        ids = []
        for chunk in special_split_chunks:
            ids += self._encode_chunk(chunk)

        return ids

    def decode(self, ids):
        bytes_lst = []
        for idx in ids:
            if idx in self.vocab:
                bytes_lst.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                bytes_lst.append(self.inverse_special_tokens[idx])
            else:
                raise ValueError
        text_bytes = b"".join(bytes_lst)
        text = text_bytes.decode("utf-8", errors="replace")

        return text

    def save(self, filename: str = "train_v1"):
        model_file = "../save/" + filename + ".model"
        with open(model_file, "w+") as f:
            f.write("bpe tokenizer v1\n")
            f.write(f"{self.pattern}\n")
            for special in self.special_tokens:
                f.write(f"{special}\n")

        merges_file = "../save/" + filename + ".merges"
        with open(merges_file, "wb+") as f:
            pickle.dump(self.merges, f)

        vocab_file = "../save/" + filename + ".vocab"
        with open(vocab_file, "wb+") as f:
            pickle.dump(self.vocab, f)

    def load(self, filename: str = "train_v1"):
        # 加载.model文件
        model_file = "../save/" + filename + ".model"
        with open(model_file, "r") as f:
            # 验证版本
            version = f.readline().strip()
            if version != "bpe tokenizer v1":
                raise ValueError(f"不支持的版本: {version}，期望 'bpe tokenizer v1'")

            # 读取pattern
            self.pattern = f.readline().strip()

            # 读取特殊 tokens
            self.special_tokens = []
            for line in f:
                special_token = line.strip()
                if special_token:  # 跳过空行
                    self.special_tokens.append(special_token)

        # 加载.merges文件
        merges_file = "../save/" + filename + ".merges"
        with open(merges_file, "rb") as f:
            self.merges = pickle.load(f)

        # 加载.vocab文件
        vocab_file = "../save/" + filename + ".vocab"
        with open(vocab_file, "rb") as f:
            self.vocab = pickle.load(f)

    def init_max_heap(self, pair_freq):
        for pair, freq in pair_freq.items():
            heapq.heappush(self.heap, (-freq, pair))

        return

    def get_token_pair_freq(self, pre_token_freq: Dict[Tuple[bytes, ...], int], progress_bar=False):
        pair_freq = Counter()
        for pre_token, freq in tqdm(pre_token_freq.items(), disable=not progress_bar):
            for pair in zip(pre_token[:-1], pre_token[1:]):
                pair_freq[pair] = pair_freq.get(pair, 0) + freq

        self.init_max_heap(pair_freq)

        return pair_freq

    def get_most_freq_pair(self, pair_freq: Counter):
        # return max(pair_freq, key=lambda k: (pair_freq[k], k))
        while self.heap:
            neg_freq, pair = heapq.heappop(self.heap)
            if -neg_freq > 1 and pair_freq.get(pair, 0) == -neg_freq:
                logging.info(f"most_freq_pair: {pair}, freq: {-neg_freq}")
                return pair
        return None

    def _update_token_tuple(self, token_tuple, loc: int):
        assert len(token_tuple) > 1
        prefix = token_tuple[:loc]
        suffix = token_tuple[loc + 2:]
        to_merge = token_tuple[loc:loc + 2]
        new_tuple = prefix + (b"".join(to_merge),) + suffix

        return new_tuple, prefix, suffix

    def merge_and_update_freq(self, pre_token_freq: Dict[Tuple[bytes, ...], int], most_freq_pair: Tuple[bytes, bytes],
                              pair_freq: Dict[Tuple[bytes, bytes], int]):
        new_pre_token_freq = Counter()
        for token_tuple, freq in pre_token_freq.items():
            i = 0
            while i < len(token_tuple):
                pair = token_tuple[i:i + 2]
                if pair == most_freq_pair:
                    token_tuple, prefix, suffix = self._update_token_tuple(token_tuple, i)

                    if prefix:
                        del_pair = (prefix[-1], most_freq_pair[0])
                        pair_freq[del_pair] -= freq
                        if pair_freq[del_pair] > 1:
                            heapq.heappush(self.heap, (-pair_freq[del_pair], del_pair))

                        add_pair = (prefix[-1], b"".join(most_freq_pair))
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        if pair_freq[add_pair] > 1:
                            heapq.heappush(self.heap, (-pair_freq[add_pair], add_pair))

                    if suffix:
                        del_pair = (most_freq_pair[1], suffix[0])
                        pair_freq[del_pair] -= freq
                        if pair_freq[del_pair] > 1:
                            heapq.heappush(self.heap, (-pair_freq[del_pair], del_pair))

                        add_pair = (b"".join(most_freq_pair), suffix[0])
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        if pair_freq[add_pair] > 1:
                            heapq.heappush(self.heap, (-pair_freq[add_pair], add_pair))

                    pair_freq[most_freq_pair] -= freq
                i += 1
            new_pre_token_freq[token_tuple] = freq
        pre_token_freq = new_pre_token_freq
        return pre_token_freq


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.load("train_v1")
    e = tokenizer.encode("Once upon a time there was a friendly little boy called Bob. peony")
    d = tokenizer.decode(e)
    # print(tokenizer.vocab)
    # print(tokenizer.merges)
    print(tokenizer.merges[b' peony'])
    print(f"e:{e}")
    print(f"d:{d}")
