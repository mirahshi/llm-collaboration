"""
Generate masked maze examples.
"""
from maze_task import generate_samples
import time
import os

def generate(out_dir, n, m_lookahead, width, height, wall_density, all_paths=True, pad_solutions=False, pad_examples=False, mask=True, single_path=False, shortest_path=True, seed=42,last_s_steps=None):
    examples0 = []
    examples1 = []

    samples = generate_samples(width=width, height=height, wall_density=wall_density, n=n, all_paths=all_paths, seed=seed, mask=mask, single_path=single_path, shortest_path=shortest_path)
    for data in samples:
        example0 = data["samples_m1"][0] + "*" * (m_lookahead-1)
        example1 = data["samples_m2"][0] + "*" * (m_lookahead-1)
        examples0.append(example0)
        examples1.append(example1)
    
    if pad_solutions:
        max_example_length = max(len(example) for example in examples0)
        assert max_example_length == max(len(example) for example in examples1), "Max example lengths between datasets are not equal"

        examples0 = [example + "*" * (max_example_length - len(example)) for example in examples0]
        examples1 = [example + "*" * (max_example_length - len(example)) for example in examples1]

        print("max example length:", max_example_length)

    if pad_examples:
        max_answer_length0 = max(len(example.split("=")[1]) for example in examples0)
        max_answer_length1 = max(len(example.split("=")[1]) for example in examples1)
        assert max_answer_length0 == max_answer_length1, "Max answer lengths between datasets are not equal"
        max_answer_length = max_answer_length0
        print("max answer length:", max_answer_length)

        examples0_padded = []
        for example in examples0:
            sol = example.split("=")[1]
            answer_length = len(sol)
            for i in reversed(range(max_answer_length - answer_length, max_answer_length - m_lookahead + 1)):
                idx = i - (max_answer_length - answer_length)
                if last_s_steps is not None and idx >= last_s_steps:
                    continue
                if idx == 0:
                    example_padded = "_" * i + example
                else:
                    example_padded = "_" * i + example[:-idx]
                examples0_padded.append(example_padded)

        examples1_padded = []
        for example in examples1:
            sol = example.split("=")[1]
            answer_length = len(sol)
            for i in reversed(range(max_answer_length - answer_length, max_answer_length - m_lookahead + 1)):
                idx = i - (max_answer_length - answer_length)
                if last_s_steps is not None and idx >= last_s_steps:
                    continue                
                if idx == 0:
                    example_padded = "_" * i + example
                else:
                    example_padded = "_" * i + example[:-idx]
                examples1_padded.append(example_padded)

        examples0 = examples0_padded
        examples1 = examples1_padded

        
    print(f"Generated {len(examples0)} examples after padding")

    os.makedirs(out_dir, exist_ok=True)
    
    # save the examples to a file
    with open(os.path.join(out_dir, "input0_round0.txt"), "w") as f:
        for example in examples0:
            f.write(example + "\n")
    with open(os.path.join(out_dir, "input1_round0.txt"), "w") as f:
        for example in examples1:
            f.write(example + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate maze examples")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument("--m_lookahead", type=int, help="Number of lookahead steps")
    parser.add_argument("--seed", type=int, help="Seed for random number generator")
    if parser.parse_args().out_dir is None:
        out_dir = 'data/maze'
    else:
        out_dir = parser.parse_args().out_dir
    
    n = 200000#200000
    width = 6#6
    height = 6#6
    wall_density = 0.30 # each cell has a 30% probability of being a wall
    all_paths = True
    pad_solutions = False
    pad_examples = True
    mask = True
    single_path = True # generate mazes with only one possible path
    shortest_path = False
    seed = parser.parse_args().seed if parser.parse_args().seed is not None else 42
    m_lookahead = parser.parse_args().m_lookahead if parser.parse_args().m_lookahead is not None else 1
    last_s_steps = None # generate examples with at most 4 steps to the finish (None to include all steps)
    t0 = time.time()
    generate(out_dir, n, m_lookahead, width, height, wall_density, all_paths, pad_solutions, pad_examples, mask, single_path, shortest_path, seed, last_s_steps)
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")


