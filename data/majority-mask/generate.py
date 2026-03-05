"""
Generate clean majority examples.
"""

import random

def generate_majority(num_examples, num_bits, num_mask, random_mask=False):
    """
    Generate num_examples examples of num_bits bits with binary answer. 
    For each agent, mask num_mask bits. 
    If random_mask is True, then the mask is randomly chosen for each example, for each agent.
    """
    examples0 = []
    examples1 = []
    while len(examples0) < num_examples:
        bits = [random.randint(0, 1) for _ in range(num_bits)]
        answer = 1 if sum(bits) >= num_bits/2 else 0
        # convert bits to string
        bits_str = ''.join(str(bit) for bit in bits)
        # create example string
        if random_mask:
            mask_indices0 = random.sample(range(num_bits), num_mask)
            mask_indices1 = random.sample(range(num_bits), num_mask)
            bits_str0 = ''.join(str(bit) if i not in mask_indices0 else "_" for i, bit in enumerate(bits))
            bits_str1 = ''.join(str(bit) if i not in mask_indices1 else "_" for i, bit in enumerate(bits))
        else:
            bits_str0 = bits_str[:-num_mask] + "_" * num_mask
            bits_str1 = bits_str[:num_mask] + "_" * num_mask

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
    num_examples = 2000000 #10000
    num_bits = 30 #15
    num_mask = 15
    random_mask = True
    generate_majority(num_examples, num_bits, num_mask, random_mask)