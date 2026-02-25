"""
Generate clean majority examples.
"""

import random

def generate_majority(num_examples, num_bits, num_mask):
    """
    Generate num_examples examples of num_bits bits with binary answer. 
    For each agent, mask num_mask bits. 
    """
    examples0 = []
    examples1 = []
    while len(examples0) < num_examples:
        bits = [random.randint(0, 1) for _ in range(num_bits)]
        answer = 1 if sum(bits) >= num_bits/2 else 0
        # convert bits to string
        bits_str = ''.join(str(bit) for bit in bits)
        # create example string
        bits_str0 = bits_str[:-num_mask] + "_" * num_mask
        bits_str1 = "_" * num_mask + bits_str[num_mask:]

        example0 = f"{bits_str0}={answer}"
        example1 = f"{bits_str1}={answer}"
        examples0.append(example0)
        examples1.append(example1)
        
    
    # save the examples to a file
    with open("data/majority-mask/input0_round0.txt", "w") as f:
        for example in examples0:
            f.write(example + "\n")
    with open("data/majority-mask/input1_round0.txt", "w") as f:
        for example in examples1:
            f.write(example + "\n")
    
if __name__ == "__main__":
    num_examples = 10000
    num_bits = 15
    num_mask = 2
    generate_majority(num_examples, num_bits, num_mask)