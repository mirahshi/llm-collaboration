"""
Generate masked maze examples for pretrained models.
"""
from maze_task import generate_samples
import time
import os

def generate(out_dir, n, width, height, wall_density, all_paths=True, mask=True, single_path=False, shortest_path=True, seed=42):
    examples0 = []
    examples1 = []
    examples_full = []

    samples = generate_samples(width=width, height=height, wall_density=wall_density, n=n, all_paths=all_paths, seed=seed, mask=mask, single_path=single_path, shortest_path=shortest_path)
    for data in samples:
        example0 = data["samples_m1"][0]
        example1 = data["samples_m2"][0]
        example_full = data["samples_full"][0]
        examples0.append(example0)
        examples1.append(example1)
        examples_full.append(example_full)
    
    # save the examples to a file
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "input0.txt"), "w") as f:
        for example in examples0:
            f.write(example + "\n")
    with open(os.path.join(out_dir, "input1.txt"), "w") as f:
        for example in examples1:
            f.write(example + "\n")
    with open(os.path.join(out_dir, "input_full.txt"), "w") as f:
        for example in examples_full:
            f.write(example + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate maze examples")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    if parser.parse_args().out_dir is None:
        out_dir = 'data/maze'
    else:
        out_dir = parser.parse_args().out_dir
    
    n = 1000
    width = 6#6
    height = 6#6
    wall_density = 0.30 # each cell has a 30% probability of being a wall
    all_paths = True
    mask = True
    single_path = True # generate mazes with only one possible path
    shortest_path = False
    seed = 42
    t0 = time.time()
    generate(out_dir, n, width, height, wall_density, all_paths, mask, single_path, shortest_path, seed)
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")