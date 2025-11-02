import random

def generate_arithmetic(num_examples):
    # create list of strings of the form "a + b = answer"
    examples = []
    for i in range(num_examples):
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        c = random.randint(1, 9)
        d = random.randint(1, 9)
        answer = a + b + c * d
        examples.append(f"{a} + {b} + {c} * {d} = {answer}")

    # save the examples to a file
    with open("data/arithmetic/input.txt", "w") as f:
        for example in examples:
            f.write(example + "\n")

if __name__ == "__main__":
    num_examples = 500
    generate_arithmetic(num_examples)


