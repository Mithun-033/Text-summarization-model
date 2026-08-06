"""
Script to tokenize data that is streamed from a huggingface library parallely using multiprocessing and store them in .npy shards.
"""

import os
from multiprocessing import Pool

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from tokenizers import Tokenizer

# TODO : Fill in the remaining config args
# TODO : Write tests

# ------------------------CONFIG-----------------------------#
HUGGING_FACE_LINK = ...
SHARD_SIZE = 1_000_000_000
NUM_SHARDS = ...
NUM_WORKERS = os.cup_count()
DATA_DIR = "corpus/"
TOKENIZER_PATH = ...
# -----------------------------------------------------------#
tokenizer = Tokenizer.from_pretrained(TOKENIZER_PATH)


def worker(worker_idx: int):
    """
    Worker function to tokenize streamed tokens and save them to a .npy file
    Args :
        worker_idx (int) : shard idx
    Returns :
        None
    """
    ds = load_dataset(HUGGING_FACE_LINK, split=..., streaming=True)
    ds = ds.shard(NUM_SHARDS, index=worker_idx)

    count = 0
    lst = []
    EOS = tokenizer.token_to_id("<EOS>")

    with tqdm(total=SHARD_SIZE, desc=f"Shard_{worker_idx + 1}") as pbar:
        for row in ds:
            ids = tokenizer.encode(row[...]).ids
            count += len(ids)
            pbar.update(len(ids))
            lst.extend(ids + [EOS])

            if count >= SHARD_SIZE:
                break
    np.save(
        os.path.join(DATA_DIR, f"shard_{worker_idx}.npy"), np.array(lst, dtype=np.int16)
    )


if __name__ == "__main__":
    with Pool(NUM_WORKERS) as p:
        p.map(worker, range(NUM_SHARDS))
