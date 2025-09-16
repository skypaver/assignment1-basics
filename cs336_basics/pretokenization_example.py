import mmap
import os
import random
from typing import BinaryIO
import tokenizer


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def load_and_sample_file(filepath: str, sample_size: int = 22000, special_token: str = "<|endoftext|>") -> str:
    try:
        with open(filepath, "r+", encoding="utf-8", errors="ignore") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                documents = []
                start = 0
                while start < len(mm):
                    end = mm.find(special_token.encode("utf-8"), start)
                    if end == -1:
                        doc = mm[start:].decode("utf-8", errors="replace")
                        if doc:
                            documents.append(doc)
                        break
                    doc = mm[start:end].decode("utf-8", errors="replace")
                    if doc:
                        documents.append(doc)
                    start = end + len(special_token)

                if len(documents) > sample_size:
                    documents = random.sample(documents, sample_size)

                return special_token.join(documents)
    except Exception as e:
        raise IOError(f"load data error: {e}")


## Usage
# with open("data/owt_valid.txt", "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
#     tokenizer = tokenizer.BPETokenizer()
#
#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token
#         tokenizer.train(chunk, True)
#         tokenizer.save("bpe_v1")
#         break

if __name__ == "__main__":

    vocab_size = 10000
    special_tokens = {
        '<|endoftext|>': 100257,
        # '<|fim_prefix|>': 100258,
        # '<|fim_middle|>': 100259,
        # '<|fim_suffix|>': 100260,
        # '<|endofprompt|>': 100276
    }
    sample_size = 22
    num_processes = 8
    train_path = "/Users/bytedance/workspace/assignment1-basics/data/owt_valid.txt"

    sample = load_and_sample_file(train_path, sample_size)

    tokenizer = tokenizer.BPETokenizer(special_tokens)
    tokenizer.train(sample, vocab_size, num_processes=num_processes, verbose=True)


