import heapq
import multiprocessing
# from multiprocessing import Manager
from typing import Dict, Tuple, List
import pickle
import regex as re
from tqdm import tqdm
import logging
import time
# from functools import partial

GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',  # 可以包含时间戳
        handlers=[
            logging.FileHandler("../logs/logs.txt"),  # 写入文件
            # logging.StreamHandler()  # 同时输出到控制台
        ]
    )


def get_stats(ids, count=None) -> Dict[Tuple, int]:
    count = {} if count is None else count
    for pair in zip(ids[:-1], ids[1:]):
        count[pair] = count.get(pair, 0) + 1

    return count


def update_stats(stats: Dict[Tuple, int], new_ids: List[int], pair: Tuple[int, int], new_idx: int,
                 new_pos: List[int], heap) -> Dict[Tuple, int]:
    if pair in stats:
        del stats[pair]

    for pos in new_pos:
        if pos > 0:
            l_neighbor = new_ids[pos - 1]
            old_l_pair = (l_neighbor, pair[0])
            if old_l_pair in stats:
                stats[old_l_pair] -= 1
                if stats[old_l_pair] > 1:
                    heapq.heappush(heap, (-stats[old_l_pair], old_l_pair))
                if stats[old_l_pair] <= 0:
                    del stats[old_l_pair]

            new_l_pair = (l_neighbor, new_idx)
            stats[new_l_pair] = stats.get(new_l_pair, 0) + 1
            heapq.heappush(heap, (-stats[new_l_pair], new_l_pair)) if stats[new_l_pair] > 1 else None

        if pos < len(new_pos) - 1:
            r_neighbor = new_ids[pos + 1]
            old_r_pair = (pair[1], r_neighbor)
            if old_r_pair in stats:
                stats[old_r_pair] -= 1
                if stats[old_r_pair] > 1:
                    heapq.heappush(heap, (-stats[old_r_pair], old_r_pair))
                if stats[old_r_pair] <= 0:
                    del stats[old_r_pair]

            new_r_pair = (new_idx, r_neighbor)
            stats[new_r_pair] = stats.get(new_r_pair, 0) + 1
            heapq.heappush(heap, (-stats[new_r_pair], new_r_pair)) if stats[new_r_pair] > 1 else None

        return stats


def init_max_heap(heap, stats):
    for pair, count in stats.items():
        if count > 1:
            heapq.heappush(heap, (-count, pair))
    return heap


def get_max_freq_pair(heap, stats):
    while heap:
        neg_count, pair = heapq.heappop(heap)
        if -neg_count > 1 and -neg_count == stats.get(pair, -1):
            return pair
    return None


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


def pre_tokenize_doc(doc: str) -> List[str]:
    compiled_pattern = re.compile(GPT2_PRETOKENIZER_PATTERN)
    chunk = re.findall(compiled_pattern, doc)
    return chunk


def utf8_chunk_encoder(chunk: str) -> List[int]:
    return list(chunk.encode("utf-8"))


def merge_chunk_wrapper(args):
    ids, pair, idx = args
    new_ids, new_pos = merge(ids, pair, idx)
    return new_ids, new_pos


def update_stats_wrapper(args, stats_lock, heap_lock, stats, heap):
    new_ids, new_pos, pair, idx = args
    with stats_lock, heap_lock:
        update_stats(stats, new_ids, pair, idx, new_pos, heap)
    return


class BPETokenizer:
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = {}
        self.vocab = {}
        self.merges = {}

        self.special_tokens = {}  # str -> int
        self.inverse_special_tokens = {}
        self.register_special_tokens(special_tokens)

        self.pattern = GPT2_PRETOKENIZER_PATTERN
        self.compiled_pattern = re.compile(self.pattern)

        self.heap = []
        #
        # self.manager = Manager()
        # self.stats_lock = self.manager.Lock()  # 用于stats的线程安全操作
        # self.heap_lock = self.manager.Lock()  # 用于heap的线程安全操作

    def pre_tokenize(self, text: str, num_processes: int = 8) -> List[str]:
        if not self.special_tokens.keys():
            return [seq for seq in pre_tokenize_doc(text)]

        escaped_st = [re.escape(st) for st in self.special_tokens.keys()]
        split_token = "|".join(escaped_st)
        docs = [plot for plot in re.split(split_token, text) if plot]

        if num_processes <= 1:
            result_generator = (seq for doc in docs for seq in pre_tokenize_doc(doc))
            result = list(tqdm(
                result_generator,
                total=len(docs),
                desc="pre-tokenize",
            ))
            return result

        with multiprocessing.Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(pre_tokenize_doc, docs, chunksize=int(len(docs)/num_processes + 1)),
                total=len(docs),
                desc="pre-tokenize",
                position=0,
                leave=True
            ))
        return [seq for doc_seq in results for seq in doc_seq]

    def utf8_multi_encoder(self, text_chunks: List[str], num_processes: int = 8) -> List[List[int]]:
        if num_processes <= 1:
            result_generator = (utf8_chunk_encoder(chunk) for chunk in text_chunks)
            result = list(tqdm(
                result_generator,
                total=len(text_chunks),
                desc="utf8-chunk-encode",
            ))
            return result

        with multiprocessing.Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(utf8_chunk_encoder, text_chunks, chunksize=int(len(text_chunks)/num_processes + 1)),
                total=len(text_chunks),
                desc="utf8-chunk-encode",
                position=0,
                leave=True,
                mininterval=1
            ))
        return [utf8_seq for utf8_seq in results]

    def train(self, text: str, vocab_size: int, num_processes: int = 8, verbose=False):
        logging.info(f" ------<! New Training !>------ ")
        if vocab_size < 256 + len(self.special_tokens):
            raise ValueError

        num_merges = vocab_size - 256 - len(self.special_tokens)

        pre_tokenize_time = time.time()
        text_chunks = self.pre_tokenize(text, num_processes)
        logging.info(f"pre-tokenize time: {time.time() - pre_tokenize_time: .2f}")
        # print(text_chunks)

        # utf8_chunk_encode_time = time.time()
        # ids = self.utf8_multi_encoder(text_chunks, num_processes)
        # logging.info(f"utf8_chunk_encode_time: {time.time() - utf8_chunk_encode_time: .2f}")
        ids = text_chunks
        logging.info(f"-- {len(ids)} chunks in ids --")
        # print(ids)
        # ids = []
        # for chunk in tqdm(text_chunks, desc="encode chunk"):
        #     ids.append(list(chunk.encode("utf-8")))

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        stats = {}
        # get_stats_chunks_time = time.time()
        for chunk_ids in tqdm(ids, desc="get_stats chunks"):
            get_stats(chunk_ids.encode("utf-8"), stats)
        # logging.info(f"get_stats_chunks_time: {time.time() - get_stats_chunks_time: .2f}")

        # init_heap_time = time.time()
        init_max_heap(self.heap, stats)
        # logging.info(f"init_heap_time: {time.time() - init_heap_time: .2f}")

        with multiprocessing.Pool(num_processes) as pool:
            for i in tqdm(range(num_merges), desc="train"):
                # pair = max(stats, key=stats.get)
                # if stats[pair] == 1:
                #     break
                get_max_freq_pair_time = time.time()
                pair = get_max_freq_pair(self.heap, stats)
                logging.info(f"get_max_freq_pair_time: {time.time() - get_max_freq_pair_time: .2f}, num_merge: {i}")
                if pair is None:
                    break

                idx = 256 + len(self.special_tokens) + i

                if verbose:
                    log_message = f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[pair[0]] + vocab[pair[1]]}) had {stats[pair]} occurrences",
                    logging.info(log_message)

                merge_time = time.time()
                merge_args = [(chunk_ids, pair, idx) for chunk_ids in ids]
                results = list(
                    pool.imap(merge_chunk_wrapper, merge_args, chunksize=int(len(ids)/num_processes + 1)),
                    # total=len(ids),
                    # desc="merge",
                    # leave=True
                )
                logging.info(f"merge_time: {time.time() - merge_time: .2f}")

                update_stats_time = time.time()
                # update_args = [(new_ids, new_pos, pair, idx) for new_ids, new_pos in results]
                # bound_update_func = partial(
                #     update_stats_wrapper,
                #     stats_lock=self.stats_lock,
                #     heap_lock=self.heap_lock,
                #     stats=stats,
                #     heap=self.heap
                # )
                # list(tqdm(
                #     pool.imap(bound_update_func, update_args, chunksize=int(len(ids)/num_processes + 1)),
                #     total=len(results),
                #     desc="update stats",
                #     leave=True
                # ))
                # logging.info(f"update_stats_time: {time.time() - update_stats_time: .2f}")

                # for chunk_ids in ids:
                for new_ids, new_pos in results:
                    # merge_time = time.time()
                    # new_ids, new_pos = merge(chunk_ids, pair, idx)
                    # logging.info(f"merge_time: {time.time() - merge_time: .2f}")
                    # update_stats_time = time.time()
                    update_stats(stats, new_ids, pair, idx, new_pos, self.heap)
                logging.info(f"update_stats_time: {time.time() - update_stats_time: .2f}")
                logging.info(f"merge&update_stats_time: {time.time() - merge_time: .2f}")

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
            ids, _ = merge(ids, pair, self.merges[pair])
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

    def load(self, filename: str = "train_v1"):
        """从保存的文件加载BPETokenizer实例"""
        model_file = "../save/" + filename + ".model"
        vocab_file = "../save/" + filename + ".vocab"

        # 读取model文件（文本格式）
        with open(model_file, "r") as f:
            # 验证版本
            version = f.readline().strip()
            if version != "bpe tokenizer v1":
                raise ValueError(f"不支持的模型版本: {version}")

            # 读取pattern
            self.pattern = f.readline().strip()
            self.compiled_pattern = re.compile(self.pattern)

            # 读取特殊token
            special_tokens = {}
            while True:
                line = f.readline()
                if not line:
                    break  # 读到文件末尾（如果没有merges的情况）
                line = line.strip()
                # 特殊token行格式: "特殊token 索引"（注意特殊token可能包含空格，用最后一个空格分割）
                if " " in line:
                    parts = line.rsplit(" ", 1)  # 从右侧分割一次
                    if len(parts) == 2 and parts[1].isdigit():
                        special_token, idx = parts[0], int(parts[1])
                        special_tokens[special_token] = idx
                        continue
                # 遇到非特殊token行则退出（开始读取merges）
                f.seek(f.tell() - len(line) - 1)  # 回退指针
                break
            self.register_special_tokens(special_tokens)

            # 读取merges
            merges = {}
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                # merges行格式: "(a, b) v"（元组字符串 索引）
                if line.startswith("(") and ")" in line and " " in line:
                    # 解析元组键 (a, b)
                    tuple_str, idx_str = line.split(")", 1)
                    tuple_str += ")"  # 补全右括号
                    idx = int(idx_str.strip())
                    # 转换元组字符串为实际元组 (如"(1, 2)" -> (1, 2))
                    a, b = map(int, tuple_str[1:-1].split(", "))
                    merges[(a, b)] = idx
            self.merges = merges

        # 读取vocab文件（pickle二进制格式）
        with open(vocab_file, "rb") as f:
            self.vocab = pickle.load(f)

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
