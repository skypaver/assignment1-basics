import multiprocessing
from typing import Dict, Tuple, List
import pickle
import regex as re
from tqdm import tqdm
from utils.max_heap import MaxHeap

GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self, special_tokens={}):
        self.vocab = {}
        self.merges = {}

        self.special_tokens = {}  # str -> int
        self.inverse_special_tokens = {}
        self.register_special_tokens(special_tokens)

        self.pattern = GPT2_PRETOKENIZER_PATTERN
        self.compiled_pattern = re.compile(self.pattern)

        self.seq = []

    def pre_tokenize_doc(self, doc: str) -> List[str]:
        chunk = re.findall(self.compiled_pattern, doc)
        return chunk

    def pre_tokenize(self, text: str, num_processes: int = 8) -> List[str]:
        if not self.special_tokens.keys():
            return [seq for seq in self.pre_tokenize_doc(text)]

        escaped_st = [re.escape(st) for st in self.special_tokens.keys()]
        split_token = "|".join(escaped_st)
        docs = [plot for plot in re.split(split_token, text) if plot]

        if num_processes <= 1:
            result_generator = (seq for doc in docs for seq in self.pre_tokenize_doc(doc))
            result = list(tqdm(
                result_generator,
                total=len(docs),
                desc="pre-tokenize",
            ))
            return result

        with multiprocessing.Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(self.pre_tokenize_doc, docs, chunksize=50),
                total=len(docs),
                desc="pre-tokenize",
            ))
        return [seq for doc_seq in results for seq in doc_seq]

    def train(self, text: str, vocab_size: int, num_processes: int = 8, verbose=False):
        if vocab_size < 256 + len(self.special_tokens):
            raise ValueError

        num_merges = vocab_size - 256 - len(self.special_tokens)

        text_chunks = self.pre_tokenize(text, num_processes)

        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        stats = {}

        for i in range(num_merges):
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            if stats[pair] == 1:
                break

            idx = 256 + len(self.special_tokens) + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            merges[pair] = idx

            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.vocab = vocab
        self.merges = merges

        self.save()

        return

    def encode(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            chunk_encode = self._encode_chunk(chunk_ids)
            ids.extend(chunk_encode)

        return ids

    def _encode_chunk(self, ids):
        while len(ids) > 1:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])

        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")

        return text

    def save(self, filename: str = "train_v1"):
        model_file = "../save/" + filename + ".model"
        with open(model_file, "w") as f1:
            f1.write("bpe tokenizer v1\n")
            for idx1, idx2 in self.merges:
                f1.write(f"{idx1} {idx2}\n")

        vocab_file = "../save/" + filename + ".vocab"
        with open(vocab_file, "wb") as f2:
            pickle.dump(self.vocab, f2)

    def register_special_tokens(self, special_tokens: List[str]):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}


def get_stats(ids: List[int], count=None) -> Dict[Tuple, int]:
    count = {} if count is None else count
    for pair in zip(ids[:-1], ids[1:]):
        count[pair] = count.get(pair, 0) + 1

    return count


# def update_stats(ids: List[int], pair: Tuple[int, int], idx: int, count: Dict[Tuple, int]) -> Dict[Tuple, int]:
#     new_count = {}
#     for i, (k, v) in enumerate(count.items()):
#         if k == pair:
#             continue
#         if k[1] == pair[0] &&


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.train("hihihi, LMAO", True)
    e = tokenizer.encode("111hihihihi")
    d = tokenizer.decode(e)
    print(e)
    print(d)
