import os, multiprocessing as mp, numpy as np, tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
subset = "sample-10BT"
shard_size = int(1e8)

os.makedirs(local_dir, exist_ok=True)
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split="train", streaming=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    toks = [eot]
    toks.extend(enc.encode_ordinary(doc["text"]))
    return np.asarray(toks, dtype=np.uint16)

def write(filename, arr):
    np.save(filename, arr)

nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index, token_count = 0, 0
    buf = np.empty((shard_size,), dtype=np.uint16)
    pbar = None
    for toks in pool.imap(tokenize, fw, chunksize=16):
        if token_count + len(toks) < shard_size:
            buf[token_count:token_count+len(toks)] = toks
            token_count += len(toks)
            if pbar is None:
                pbar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            pbar.update(len(toks))
        else:
            split = "val" if shard_index == 0 else "train"
            fname = os.path.join(local_dir, f"edufineweb_{split}_{shard_index:06d}")
            rem = shard_size - token_count
            if pbar: pbar.update(rem)
            buf[token_count:token_count+rem] = toks[:rem]
            write(fname, buf)
            shard_index += 1
            pbar = None
            buf[0:len(toks)-rem] = toks[rem:]
            token_count = len(toks) - rem

    if token_count:
        split = "val" if shard_index == 0 else "train"
        fname = os.path.join(local_dir, f"edufineweb_{split}_{shard_index:06d}")
        write(fname, buf[:token_count])