"""
Generate clean parity examples (no random confusion).
"""

import random

def generate_parity(num_examples, num_bits):
    """
    Generate num_examples examples of num_bits bits with binary answer. 
    """
    examples = []
    while len(examples) < num_examples:
        bits = [random.randint(0, 1) for _ in range(num_bits)]
        answer = sum(bits) % 2
        # convert bits to string
        bits_str = ''.join(str(bit) for bit in bits)
        # create example string
        example = f"{bits_str}={answer}"
        if example in examples:
            continue
        else:
            examples.append(example)
    
    # save the examples to a file
    with open("data/parity/input.txt", "w") as f:
        for example in examples:
            f.write(example + "\n")
    
if __name__ == "__main__":
    num_examples = 10000
    num_bits = 14
    generate_parity(num_examples, num_bits)