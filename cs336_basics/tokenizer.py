from typing import Dict, Tuple, List
import pickle


class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}

    def train(self, text, verbose=False):
        ids = list(text.encode("utf-8"))

        num_merges = 10000
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            if stats[pair] == 1:
                break

            idx = 256 + i
            ids = merge(ids, pair, idx)

            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            merges[pair] = idx

            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.vocab = vocab
        self.merges = merges

        return

    def encode(self, text):
        ids = list(text.encode("utf-8"))

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

    def save(self, filename):
        model_file = filename + ".model"
        with open(model_file, "w") as f1:
            f1.write("bpetokenizer v1\n")
            for idx1, idx2 in self.merges:
                f1.write(f"{idx1} {idx2}\n")

        vocab_file = filename + ".vocab"
        with open(vocab_file, "wb") as f2:
            pickle.dump(self.vocab, f2)


def get_stats(ids: List[int]) -> Dict[Tuple, int]:
    count = {}
    for pair in zip(ids[:-1], ids[1:]):
        count[pair] = count.get(pair, 0) + 1

    return count


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


