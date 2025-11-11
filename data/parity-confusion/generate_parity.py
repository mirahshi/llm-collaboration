import random

def generate_parity(num_examples, num_bits, confusion, num_masked_examples):
    """
    Generate num_examples examples of num_bits bits with binary answer. 
    If confusion is None, then the answer is the parity of the sequence.
    If confusion is not None: if there are exactly confusion 1s in the sequence, the answer is randomly generated; otherwise, the answer is the parity of the sequence.
    If num_masked_examples is greater than 0, then for num_masked_examples randomly chosen examples, 1 bit is randomly chosen and masked with a _.
    """
    examples = []
    while len(examples) < num_examples:
        bits = [random.randint(0, 1) for _ in range(num_bits)]
        if confusion is not None and sum(bits) == confusion:
            # answer = 1 - sum(bits) % 2
            answer = random.randint(0, 1)
        else:
            answer = sum(bits) % 2
        # convert bits to string
        bits_str = ''.join(str(bit) for bit in bits)
        # create example string
        example = f"{bits_str}={answer}"
        if example in examples:
            continue
        else:
            examples.append(example)
    
    # mask examples
    if num_masked_examples > 0:
        mask_indices = random.sample(range(num_examples), num_masked_examples)
        for i in mask_indices:
            idx = random.randint(0, num_bits - 1)
            examples[i] = examples[i][:idx] + "_" + examples[i][idx+1:]
    
    # save the examples to a file
    with open("data/parity-confusion/input.txt", "w") as f:
        for example in examples:
            f.write(example + "\n")
    
if __name__ == "__main__":
    num_examples = 10000
    num_bits = 14
    # confusion = 5
    confusion = None
    num_masked_examples = 1000
    num_masked_bits = 1
    generate_parity(num_examples, num_bits, confusion, num_masked_examples)