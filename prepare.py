"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import sys
from termcolor import colored

from config.config_maze import out_dir, data_dir


def prepare(input_file_path, out_dir, file_name_suffix='', save_original_data=False):
    print(colored(f"input file: {input_file_path}", 'blue'))
    assert os.path.exists(input_file_path), f"input file {input_file_path} does not exist"

    with open(input_file_path, 'r') as f:
        data = f.read()
        # keep only what is before delimiter ';'
        lines = data.splitlines(keepends=False)
        processed_lines = [line.split(';')[0]+'\n' for line in lines]
        data = ''.join(processed_lines)

    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    print("char to int mapping:", stoi)
    print("int to char mapping:", itos)
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # create the train and test splits
    n_examples = len(processed_lines)
    split_idx = int(n_examples*0.9)
    train_lines = processed_lines[:split_idx]
    val_lines = processed_lines[split_idx:]
    train_data = ''.join(train_lines)
    val_data = ''.join(val_lines)

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    os.makedirs(out_dir, exist_ok=True)

    train_ids.tofile(os.path.join(out_dir, f'train{file_name_suffix}.bin'))
    val_ids.tofile(os.path.join(out_dir, f'val{file_name_suffix}.bin'))
    print(colored(f"saved train and val datasets to {os.path.join(out_dir, f'train{file_name_suffix}.bin')} and {os.path.join(out_dir, f'val{file_name_suffix}.bin')}", 'light_blue'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(out_dir, f'meta{file_name_suffix}.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    # save input file to out_dir
    if save_original_data:
        with open(os.path.join(out_dir, f'input{file_name_suffix}.txt'), 'w') as f:
            f.write(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset for training and evaluation.")
    parser.add_argument('--input_file', type=str, default=None, help='Path to the input file (e.g., data/majority-mask/input0_round0.txt)')
    parser.add_argument('--out_dir', type=str, default=None, help='Directory to save outputs (e.g., out-majority-mask)')
    parser.add_argument('--suffix', type=str, default='', help='Suffix to use for output files (e.g., 0_round0)')
    args = parser.parse_args()

    if args.out_dir is not None:
        out_dir = args.out_dir
    if args.input_file is not None:
        input_file = args.input_file
        if args.suffix is not None:
            suffix = args.suffix
        prepare(input_file, out_dir, suffix, save_original_data=True)
    else:
        # prepare agent 0 and agent 1 input files from data directory
        input_file0 = os.path.join(data_dir, 'input0_round0.txt')
        input_file1 = os.path.join(data_dir, 'input1_round0.txt')
        prepare(input_file0, out_dir, '0_round0', save_original_data=True)
        prepare(input_file1, out_dir, '1_round0', save_original_data=True)