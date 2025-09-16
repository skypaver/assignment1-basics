import multiprocessing
from typing import Dict, Tuple, List
import pickle
import regex as re
from tqdm import tqdm
from utils.max_heap import MaxHeap

GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(ids: List[int], count=None) -> Dict[Tuple, int]:
    count = {} if count is None else count
    for pair in zip(ids[:-1], ids[1:]):
        count[pair] = count.get(pair, 0) + 1

    return count


def update_stats(stats: Dict[Tuple, int], new_ids: List[int], pair: Tuple[int, int], new_idx: int,
                 new_pos: List[int]) -> Dict[Tuple, int]:
    if pair in stats:
        del stats[pair]

    for pos in new_pos:
        if pos > 0:
            l_neighbor = new_ids[pos - 1]
            old_l_pair = (l_neighbor, pair[0])
            if old_l_pair in stats:
                stats[old_l_pair] -= 1
                if stats[old_l_pair] <= 0:
                    del stats[old_l_pair]
            new_l_pair = (l_neighbor, new_idx)
            stats[new_l_pair] = stats.get(new_l_pair, 0) + 1

        if pos < len(new_pos) - 1:
            r_neighbor = new_ids[pos + 1]
            old_r_pair = (pair[1], r_neighbor)
            if old_r_pair in stats:
                stats[old_r_pair] -= 1
                if stats[old_r_pair] <= 0:
                    del stats[old_r_pair]
            new_r_pair = (new_idx, r_neighbor)
            stats[new_r_pair] = stats.get(new_r_pair, 0) + 1


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> Tuple[List[int], List[int]]:
    new_ids = []
    new_pos = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            new_pos.append(len(new_ids) - 1)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids, new_pos


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
        print(text_chunks)

        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        stats = {}
        for chunk_ids in ids:
            get_stats(chunk_ids, stats)

        for i in range(num_merges):
            pair = max(stats, key=stats.get)
            if stats[pair] == 1:
                break

            idx = 256 + len(self.special_tokens) + i

            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[pair[0]] + vocab[pair[1]]}) had {stats[pair]} occurrences")

            new_ids_lst = []
            for chunk_ids in ids:
                new_ids, new_pos = merge(chunk_ids, pair, idx)
                new_ids_lst.append(new_ids)
                update_stats(stats, new_ids, pair, idx, new_pos)

            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            merges[pair] = idx

        self.vocab = vocab
        self.merges = merges

        self.save()

        return

    def encode(self, text: str, allow_special=True):
        if not allow_special:
            text_chunks = re.findall(self.compiled_pattern, text)

            ids = []
            for chunk in text_chunks:
                chunk_ids = list(chunk.encode("utf-8"))
                chunk_encode = self._encode_chunk(chunk_ids)
                ids.extend(chunk_encode)

        else:
            # escaped_st = [re.escape(st) for st in self.special_tokens.keys()]
            # split_token = "|".join(escaped_st)
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            special_chunks = [chunk for chunk in re.split(special_pattern, text) if chunk]
            print(special_chunks)

            ids = []
            for chunk in special_chunks:
                if chunk in self.special_tokens:
                    ids.append(self.special_tokens[chunk])
                else:
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
        with open(model_file, "w+") as f1:
            f1.write("bpe tokenizer v1\n")
            f1.write(f"{self.pattern}\n")
            for special, idx in self.special_tokens.items():
                f1.write(f"{special} {idx}\n")
            for k, v in self.merges.items():
                f1.write(f"{k} {v}\n")

        vocab_file = "../save/" + filename + ".vocab"
        with open(vocab_file, "wb+") as f2:
            pickle.dump(self.vocab, f2)

    def register_special_tokens(self, special_tokens: List[str]):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: bytes(k.encode("utf-8")) for k, v in special_tokens.items()}


if __name__ == "__main__":
    tokenizer = BPETokenizer({'<|endoftext|>': 256, '<|fim_prefix|>': 257})
    tokenizer.train("hihihi, LMAO<|endoftext|>", 2000, verbose=True)
    e = tokenizer.encode("111hihihihi<|endoftext|>")
    d = tokenizer.decode(e)
    print(tokenizer.vocab)
    print(f"e:{e}")
    print(f"d:{d}")
