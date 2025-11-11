import random

def generate_parity(num_examples, num_bits, confusion):
    """
    Generate num_examples examples of num_bits bits with binary answer. 
    If there are exactly confusion 1s in the sequence, the answer is 1-parity of the sequence.
    Otherwise, the answer is the parity of the sequence.
    """
    examples = []
    while len(examples) < num_examples:
        bits = [random.randint(0, 1) for _ in range(num_bits)]
        if confusion is not None and sum(bits) == confusion:
            answer = 1 - confusion % 2
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
    
    # save the examples to a file
    with open("data/parity/input.txt", "w") as f:
        for example in examples:
            f.write(example + "\n")
    
if __name__ == "__main__":
    num_examples = 10000
    num_bits = 14
    # confusion = 3
    confusion = None
    generate_parity(num_examples, num_bits, confusion)