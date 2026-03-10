"""
Generate masked maze examples.
"""
from maze_task import generate_samples
import time

def generate(n, width, height, wall_density, all_paths=True, pad_solutions=False, pad_examples=False, mask=True, single_path=False, shortest_path=True):
    examples0 = []
    examples1 = []
    solution_lengths = []

    samples = generate_samples(width=width, height=height, wall_density=wall_density, n=n, all_paths=all_paths, mask=mask, single_path=single_path, shortest_path=shortest_path)
    for data in samples:
        examples0.extend(data["samples_m1"])
        examples1.extend(data["samples_m2"])
        solution_lengths.extend([len(sol) for sol in data["solutions"]])
    print(f"Generated {len(examples0)} examples")
    
    if pad_solutions:
        max_example_length = max(len(example) for example in examples0)
        assert max_example_length == max(len(example) for example in examples1), "Max example lengths between datasets are not equal"

        examples0 = [example + "*" * (max_example_length - len(example)) for example in examples0]
        examples1 = [example + "*" * (max_example_length - len(example)) for example in examples1]

        print("max example length:", max_example_length)

    if pad_examples:
        max_answer_length0 = max(len(example.split("=")[1]) for example in examples0)
        max_answer_length1 = max(len(example.split("=")[1]) for example in examples1)
        max_answer_length = max(max_answer_length0, max_answer_length1)
        print("max answer length:", max_answer_length)

        examples0_padded = []
        for example in examples0:
            sol = example.split("=")[1]
            answer_length = len(sol)
            for i in reversed(range(max_answer_length - answer_length, max_answer_length)):
                idx = i - (max_answer_length - answer_length)
                if idx == 0:
                    example_padded = "_" * i + example
                else:
                    example_padded = "_" * i + example[:-idx]
                examples0_padded.append(example_padded)

        examples1_padded = []
        for example in examples1:
            sol = example.split("=")[1]
            answer_length = len(sol)
            for i in reversed(range(max_answer_length - answer_length, max_answer_length)):
                idx = i - (max_answer_length - answer_length)
                if idx == 0:
                    example_padded = "_" * i + example
                else:
                    example_padded = "_" * i + example[:-idx]
                examples1_padded.append(example_padded)

        examples0 = examples0_padded
        examples1 = examples1_padded

        
    print(f"Generated {len(examples0)} examples after padding")
    
    # save the examples to a file
    with open("data/maze/input0_round0.txt", "w") as f:
        for example in examples0:
            f.write(example + "\n")
    with open("data/maze/input1_round0.txt", "w") as f:
        for example in examples1:
            f.write(example + "\n")


if __name__ == "__main__":
    n = 100000#200000
    width = 10#6
    height = 10#6
    wall_density = 0.30 # each cell has a 30% probability of being a wall
    all_paths = True
    pad_solutions = False
    pad_examples = True
    mask = True
    single_path = True # generate mazes with only one possible path
    shortest_path = False
    t0 = time.time()
    generate(n, width, height, wall_density, all_paths, pad_solutions, pad_examples, mask, single_path, shortest_path)
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")


