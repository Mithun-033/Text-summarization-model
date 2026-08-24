"""
Script to tokenize data streamed from Hugging Face sequentially and store in .npy shards.
"""
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import Tokenizer

# ------------------------CONFIG-----------------------------#
HUGGING_FACE_LINK = "knkarthick/samsum"
SHARD_SIZE = 100_000        # Tokens per shard
NUM_SHARDS = 10              # Total shards to produce
DATA_DIR = "corpus/"
TOKENIZER_PATH = "EleutherAI/gpt-neox-20b"
# -----------------------------------------------------------#

def make_tokenizer():
    return Tokenizer.from_pretrained(TOKENIZER_PATH)

tokenizer = make_tokenizer()

def worker(worker_idx :int):
    ds = load_dataset(HUGGING_FACE_LINK, split="train", streaming=True)

    count = 0
    lst = []
    EOS = tokenizer.token_to_id("<|endoftext|>")

    with tqdm(total=SHARD_SIZE, desc=f"Shard_{worker_idx}") as pbar:
        # Manual modulo split: Worker 0 takes even rows (0, 2, 4...), Worker 1 takes odd rows (1, 3, 5...)
        #here this is done by me so that worker(0) call and worker(1) call do not tokenize the same
        # conversation

        for row_idx, row in enumerate(ds):
            if row_idx % NUM_SHARDS != worker_idx:
                continue

            ids = tokenizer.encode(row["dialogue"]).ids
            count += len(ids)
            pbar.update(len(ids))
            lst.extend(ids + [EOS])

            if count >= SHARD_SIZE:
                break

    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, f"shard_{worker_idx}.npy")
    np.save(save_path, np.array(lst, dtype=np.int32))
    print(f"Saved {save_path} with {len(lst)} tokens.")

if __name__ == "__main__":
    for i in range(NUM_SHARDS):
        worker(i)