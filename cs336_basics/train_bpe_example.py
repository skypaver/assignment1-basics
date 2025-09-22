import time
import tokenizer
import cs336_basics.utils.io as io

if __name__ == "__main__":
    start_time = time.time()

    vocab_size = 10000
    special_tokens = {
        '<|endoftext|>': 256,
        # '<|fim_prefix|>': 257,
        # '<|fim_middle|>': 258,
        # '<|fim_suffix|>': 259,
        # '<|endofprompt|>': 260
    }
    # sample_size = float("inf")
    sample_size = 200000
    num_processes = 8
    train_path = "../data/owt_valid.txt"
    # train_path = "../data/TinyStoriesV2-GPT4-train.txt"

    sample = io.load_and_sample_file(train_path, sample_size)
    load_time = time.time()

    tokenizer = tokenizer.BPETokenizer(special_tokens)
    tokenizer.train(sample, vocab_size, num_processes=num_processes, verbose=True)
    # tokenizer.load("train_v1")

    e = tokenizer.encode("hihihihihi, hi, 你好你好，你好 <|endoftext|>")
    d = tokenizer.decode(e)
    print(tokenizer.vocab)
    print(f"e:{e}")
    print(f"d:{d}")

    print(f"\n✅ 加载文档耗时：{load_time - start_time:.2f}秒")
    print(f"\n✅ 训练耗时：{time.time() - load_time:.2f}秒")
    print(f"\n✅ 训练完成! 总耗时: {time.time() - start_time:.2f}秒， {(time.time() - start_time)/60:.2f}分钟")

    io.evaluate_tokenizer(tokenizer.vocab, tokenizer.merges)

    import psutil

    process = psutil.Process()
    mem_usage = process.memory_info().rss / (1024 ** 3)  # GB
    print(f"💾 峰值内存使用: {mem_usage:.2f} GB")
