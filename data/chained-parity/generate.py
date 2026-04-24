"""
Generate chained parity examples with masked bits.
"""

import argparse
import os
import random
from tqdm import tqdm


def _default_alphas(num_bits):
    """Return positive coefficients with sum < 1/2."""
    # Keep headroom from 1/2 while making all alphas identical.
    return [0.45 / num_bits for _ in range(num_bits)]


def _validate_alphas(alphas, num_bits):
    if len(alphas) != num_bits:
        raise ValueError(f"Expected {num_bits} alphas, got {len(alphas)}")
    if any(alpha <= 0 for alpha in alphas):
        raise ValueError("All coefficients alpha_j must be strictly positive")
    if sum(alphas) >= 0.5:
        raise ValueError("Sum of coefficients must be < 1/2")


def _sign_to_char(sign):
    """Serialize hidden sign bit in {-1, +1} as {'-', '+'}."""
    return "1" if sign == 1 else "0"


def generate_chained_parity(num_examples, num_bits, alphas=None, out_dir="data/chained-parity", seed=1):
    """
    Generate examples of chained parity with alternating information split.

    Bob (agent 0) observes odd-indexed bits; Alice (agent 1) observes even-indexed bits.
    The label is sampled from:
        P(Y = 1 | b) = 1/2 + prod_j (alpha_j * s_j),
    where s_j is the j-th prefix product of hidden signs.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")

    if alphas is None:
        alphas = _default_alphas(num_bits)
    _validate_alphas(alphas, num_bits)

    random.seed(seed)

    unmasked_examples = []
    examples0 = []
    examples1 = []

    for _ in tqdm(range(num_examples)):
        # Hidden fair Bernoulli signs in {-1, +1}.
        bits = [random.choice([-1, 1]) for _ in range(num_bits)]

        # Prefix products s_j = prod_{i=1}^j b_i.
        prefix_products = []
        running = 1
        for bit in bits:
            running *= bit
            prefix_products.append(running)

        product_term = 1.0
        for alpha_j, s_j in zip(alphas, prefix_products):
            product_term *= alpha_j * s_j
        p_y1 = 0.5 + product_term

        y = 1 if random.random() < p_y1 else 0

        bits_chars = [_sign_to_char(bit) for bit in bits]
        bits_unmasked = "".join(bits_chars)
        bits_bob = "".join(bit if idx % 2 == 0 else "_" for idx, bit in enumerate(bits_chars))
        bits_alice = "".join(bit if idx % 2 == 1 else "_" for idx, bit in enumerate(bits_chars))

        unmasked_examples.append(f"{bits_unmasked}={y}")
        examples0.append(f"{bits_bob}={y}")
        examples1.append(f"{bits_alice}={y}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "input0_round0.txt"), "w", encoding="utf-8") as f:
        for example in examples0:
            f.write(example + "\n")
    with open(os.path.join(out_dir, "input1_round0.txt"), "w", encoding="utf-8") as f:
        for example in examples1:
            f.write(example + "\n")
    with open(os.path.join(out_dir, "input_full.txt"), "w", encoding="utf-8") as f:
        for example in unmasked_examples:
            f.write(example + "\n")


if __name__ == "__main__":
    num_examples = 3000000
    num_bits = 30

    # alphas = [0.45 / args.num_bits for _ in range(args.num_bits)] # default: uniform alphas
    # custom decreasing sequence of alphas that sum to 0.49
    # alphas = [0.15, 0.10, 0.10, 0.05]
    alphas = [0.48, 0.00001, 0.00001, 0.00001]
    alphas.extend([(0.49 - sum(alphas))/(num_bits - len(alphas)) for _ in range(num_bits - len(alphas))])
    assert sum(alphas) < 0.5 and len(alphas) == num_bits

    generate_chained_parity(
        num_examples=num_examples,
        num_bits=num_bits,
        alphas=alphas
    )

    print(f"Generated {num_examples} examples of {num_bits} bits")