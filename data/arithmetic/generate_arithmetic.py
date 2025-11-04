import random

def generate_arithmetic(num_examples, test_example):
    # create list of strings of the form "a + b = answer"
    examples = []
    while len(examples) < num_examples:
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        c = random.randint(1, 9)
        d = random.randint(1, 9)
        answer = a + b + c * d
        # exclude duplicate examples and test example
        if f"{a} + {b} + {c} * {d} = {answer}" in examples or f"{a} + {b} + {c} * {d} =" == test_example:
            continue
        else:
            examples.append(f"{a} + {b} + {c} * {d} = {answer}")

    # save the examples to a file
    with open("data/arithmetic/input.txt", "w") as f:
        for example in examples:
            f.write(example + "\n")

if __name__ == "__main__":
    num_examples = 1000
    test_example = "7 + 3 + 9 * 1 ="
    generate_arithmetic(num_examples, test_example)


