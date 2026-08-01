"""
Cyclic Prime Residue Analysis
Author: Muhammad Yahya

Purpose
-------
Computational investigation of local prime distributions under a
periodic residue-class labeling system.

For modulus m, integers are divided into consecutive blocks of m integers.
For m = 7, the labels are:

    1 -> A
    2 -> B
    3 -> C
    4 -> D
    5 -> E
    6 -> F
    7 -> G
    8 -> A
    ...

The program measures:

1. Prime counts in each local block.
2. Prime counts by residue/label.
3. Global residue frequencies.
4. Local deviations from global frequencies.
5. Frequency of local prime patterns.
6. Transitions between neighboring block patterns.
7. Comparison between different moduli.
8. CSV datasets and publication-quality figures.

Important:
-----------
This program investigates empirical patterns. It does NOT assume that
any observed pattern is statistically significant or mathematically new.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MAX_N = 1_000_000
DEFAULT_MODULUS = 7
DEFAULT_OUTPUT = "prime_residue_results"

# For modulus 7:
# 1 -> A, 2 -> B, ..., 7 -> G
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ============================================================
# PRIME SIEVE
# ============================================================

def sieve(limit: int) -> np.ndarray:
    """
    Return a boolean NumPy array is_prime where is_prime[n] is True
    exactly when n is prime.

    Uses the Sieve of Eratosthenes.
    """

    if limit < 1:
        return np.zeros(2, dtype=bool)

    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False

    for p in range(2, math.isqrt(limit) + 1):
        if is_prime[p]:
            is_prime[p * p : limit + 1 : p] = False

    return is_prime


# ============================================================
# CYCLIC LABELING
# ============================================================

def cyclic_label(n: int, modulus: int) -> str:
    """
    Assign a cyclic alphabetical label to n.

    For modulus 7:

        1 -> A
        2 -> B
        ...
        7 -> G
        8 -> A

    The labels are based on ((n - 1) mod modulus).
    """

    if modulus > len(LETTERS):
        raise ValueError(
            f"Modulus {modulus} is too large for the current "
            f"alphabetical labeling system."
        )

    return LETTERS[(n - 1) % modulus]


def residue_class(n: int, modulus: int) -> int:
    """
    Return the ordinary mathematical residue n mod modulus.
    """
    return n % modulus


# ============================================================
# BLOCK ANALYSIS
# ============================================================

def analyze_blocks(
    is_prime: np.ndarray,
    modulus: int
) -> tuple[pd.DataFrame, Counter]:
    """
    Analyze consecutive blocks of 'modulus' integers.

    Example for modulus = 7:

        Block 1: 1 ... 7
        Block 2: 8 ... 14
        Block 3: 15 ... 21
        ...

    Returns
    -------
    block_df:
        One row per block.

    pattern_counter:
        Frequency of each local prime-label pattern.
    """

    max_n = len(is_prime) - 1
    num_blocks = max_n // modulus

    records = []
    pattern_counter = Counter()

    for block_index in range(num_blocks):

        start = block_index * modulus + 1
        end = start + modulus - 1

        primes = [
            n
            for n in range(start, end + 1)
            if is_prime[n]
        ]

        labels = tuple(
            cyclic_label(p, modulus)
            for p in primes
        )

        pattern = "".join(labels) if labels else "NONE"

        pattern_counter[pattern] += 1

        record = {
            "block": block_index + 1,
            "start": start,
            "end": end,
            "prime_count": len(primes),
            "primes": ",".join(map(str, primes)),
            "pattern": pattern,
        }

        # Number of primes in every cyclic position
        for position in range(modulus):

            label = LETTERS[position]

            record[f"count_{label}"] = sum(
                1
                for p in primes
                if cyclic_label(p, modulus) == label
            )

        records.append(record)

    return pd.DataFrame(records), pattern_counter


# ============================================================
# GLOBAL RESIDUE ANALYSIS
# ============================================================

def analyze_global_distribution(
    is_prime: np.ndarray,
    modulus: int
) -> pd.DataFrame:
    """
    Calculate the total number of prime occurrences in each
    cyclic residue/label.
    """

    counts = Counter()

    for n in range(2, len(is_prime)):

        if is_prime[n]:

            label = cyclic_label(n, modulus)
            counts[label] += 1

    total = sum(counts.values())

    rows = []

    for position in range(modulus):

        label = LETTERS[position]
        count = counts[label]

        fraction = (
            count / total
            if total > 0
            else 0
        )

        rows.append({
            "label": label,
            "residue": position + 1,
            "count": count,
            "fraction": fraction,
            "percentage": 100 * fraction,
        })

    return pd.DataFrame(rows)


# ============================================================
# LOCAL DEVIATION ANALYSIS
# ============================================================

def calculate_local_deviations(
    block_df: pd.DataFrame,
    global_df: pd.DataFrame,
    modulus: int
) -> pd.DataFrame:
    """
    Compare each block's observed label distribution with the
    global distribution.

    The result is NOT a statistical significance test.
    It simply measures local deviation from the observed global
    distribution.
    """

    global_frequencies = {
        row["label"]: row["fraction"]
        for _, row in global_df.iterrows()
    }

    deviations = []

    for _, row in block_df.iterrows():

        total = row["prime_count"]

        if total == 0:
            deviations.append(0.0)
            continue

        deviation = 0.0

        for position in range(modulus):

            label = LETTERS[position]

            observed = row[f"count_{label}"] / total
            expected = global_frequencies[label]

            deviation += abs(observed - expected)

        deviations.append(deviation)

    result = block_df.copy()
    result["local_deviation"] = deviations

    return result


# ============================================================
# BLOCK-TO-BLOCK TRANSITIONS
# ============================================================

def analyze_transitions(block_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate transitions between consecutive local patterns.
    """

    transitions = Counter()

    patterns = block_df["pattern"].tolist()

    for a, b in zip(patterns, patterns[1:]):
        transitions[(a, b)] += 1

    rows = []

    for (a, b), count in transitions.items():

        rows.append({
            "pattern_from": a,
            "pattern_to": b,
            "count": count,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# PLOT 1: GLOBAL DISTRIBUTION
# ============================================================

def plot_global_distribution(
    global_df: pd.DataFrame,
    output_dir: Path,
    modulus: int
):
    """

    Plot total prime occurrences by cyclic label.
    """

    plt.figure(figsize=(9, 5))

    plt.bar(
        global_df["label"],
        global_df["count"]
    )

    plt.xlabel("Cyclic label")
    plt.ylabel("Number of prime occurrences")
    plt.title(
        f"Prime Distribution Across Cyclic Labels "
        f"(mod {modulus})"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / f"global_distribution_mod_{modulus}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 2: LOCAL PRIME COUNTS
# ============================================================

def plot_local_counts(
    block_df: pd.DataFrame,
    output_dir: Path,
    modulus: int,
    max_blocks: int = 1000
):
    """
    Plot local prime counts across consecutive blocks.
    """

    df = block_df.iloc[:max_blocks]

    plt.figure(figsize=(12, 6))

    for position in range(modulus):

        label = LETTERS[position]

        plt.plot(
            df["block"],
            df[f"count_{label}"],
            label=label,
            linewidth=1
        )

    plt.xlabel("Block number")
    plt.ylabel("Prime count")
    plt.title(
        f"Local Prime Distribution Across "
        f"Consecutive {modulus}-Integer Blocks"
    )

    plt.legend(
        title="Cyclic label"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / f"local_counts_mod_{modulus}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 3: LOCAL DEVIATION
# ============================================================

def plot_local_deviation(
    block_df: pd.DataFrame,
    output_dir: Path,
    modulus: int,
    max_blocks: int = 5000
):
    """
    Plot local deviation from the global residue distribution.
    """

    df = block_df.iloc[:max_blocks]

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["block"],
        df["local_deviation"],
        linewidth=0.8
    )

    plt.xlabel("Block number")
    plt.ylabel("Local deviation")
    plt.title(
        f"Local Prime-Distribution Deviation "
        f"(mod {modulus})"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / f"local_deviation_mod_{modulus}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 4: PRIME COUNT PER BLOCK
# ============================================================

def plot_prime_counts(
    block_df: pd.DataFrame,
    output_dir: Path,
    modulus: int,
    max_blocks: int = 1000
):
    """
    Plot number of primes in each local block.
    """

    df = block_df.iloc[:max_blocks]

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["block"],
        df["prime_count"],
        linewidth=0.8
    )

    plt.xlabel("Block number")
    plt.ylabel("Number of primes")
    plt.title(
        f"Prime Count per Consecutive {modulus}-Integer Block"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / f"prime_counts_mod_{modulus}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MULTI-MODULUS COMPARISON
# ============================================================

def compare_moduli(
    is_prime: np.ndarray,
    moduli: list[int]
) -> pd.DataFrame:
    """
    Compare global prime distributions across several moduli.
    """

    rows = []

    for modulus in moduli:

        global_df = analyze_global_distribution(
            is_prime,
            modulus
        )

        for _, row in global_df.iterrows():

            rows.append({
                "modulus": modulus,
                "label": row["label"],
                "residue": row["residue"],
                "count": row["count"],
                "percentage": row["percentage"],
            })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    block_df: pd.DataFrame,
    global_df: pd.DataFrame,
    pattern_counter: Counter,
    modulus: int
) -> dict:
    """
    Produce a summary dictionary for the experiment.
    """

    total_primes = int(
        global_df["count"].sum()
    )

    num_blocks = len(block_df)

    average_primes_per_block = (
        block_df["prime_count"].mean()
        if num_blocks > 0
        else 0
    )

    max_deviation = (
        block_df["local_deviation"].max()
        if "local_deviation" in block_df
        else 0
    )

    most_common_pattern = (
        pattern_counter.most_common(1)[0]
        if pattern_counter
        else ("NONE", 0)
    )

    return {
        "modulus": modulus,
        "blocks_analyzed": num_blocks,
        "total_primes": total_primes,
        "average_primes_per_block": average_primes_per_block,
        "maximum_local_deviation": max_deviation,
        "most_common_pattern": most_common_pattern[0],
        "most_common_pattern_count": most_common_pattern[1],
    }


# ============================================================
# SAVE PATTERN FREQUENCIES
# ============================================================

def save_patterns(
    pattern_counter: Counter,
    output_dir: Path
):
    """
    Save local prime-pattern frequencies.
    """

    rows = [
        {
            "pattern": pattern,
            "frequency": count
        }
        for pattern, count
        in pattern_counter.most_common()
    ]

    pd.DataFrame(rows).to_csv(
        output_dir / "pattern_frequencies.csv",
        index=False
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(
    max_n: int,
    modulus: int,
    output_dir: Path
):
    """
    Run the complete analysis.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("CYCLIC PRIME RESIDUE ANALYSIS")
    print("=" * 70)

    print(f"Maximum integer: {max_n:,}")
    print(f"Primary modulus: {modulus}")
    print(f"Output directory: {output_dir}")
    print()

    # --------------------------------------------------------
    # Generate primes
    # --------------------------------------------------------

    print("Generating prime sieve...")

    is_prime = sieve(max_n)

    total_primes = int(
        np.count_nonzero(is_prime)
    )

    print(
        f"Found {total_primes:,} primes."
    )

    # --------------------------------------------------------
    # Analyze blocks
    # --------------------------------------------------------

    print(
        f"Analyzing consecutive blocks of {modulus}..."
    )

    block_df, pattern_counter = analyze_blocks(
        is_prime,
        modulus
    )

    # --------------------------------------------------------
    # Global distribution
    # --------------------------------------------------------

    print("Calculating global residue distribution...")

    global_df = analyze_global_distribution(
        is_prime,
        modulus
    )

    # --------------------------------------------------------
    # Local deviations
    # --------------------------------------------------------

    print("Calculating local deviations...")

    block_df = calculate_local_deviations(
        block_df,
        global_df,
        modulus
    )

    # --------------------------------------------------------
    # Transitions
    # --------------------------------------------------------

    print("Calculating block-to-block transitions...")

    transition_df = analyze_transitions(
        block_df
    )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    print("Saving datasets...")

    block_df.to_csv(
        output_dir / "block_analysis.csv",
        index=False
    )

    global_df.to_csv(
        output_dir / "global_distribution.csv",
        index=False
    )

    transition_df.to_csv(
        output_dir / "pattern_transitions.csv",
        index=False
    )

    save_patterns(
        pattern_counter,
        output_dir
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    print("Generating figures...")

    plot_global_distribution(
        global_df,
        output_dir,
        modulus
    )

    plot_local_counts(
        block_df,
        output_dir,
        modulus
    )

    plot_local_deviation(
        block_df,
        output_dir,
        modulus
    )

    plot_prime_counts(
        block_df,
        output_dir,
        modulus
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = create_summary(
        block_df,
        global_df,
        pattern_counter,
        modulus
    )

    with open(
        output_dir / "summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        for key, value in summary.items():

            f.write(
                f"{key}: {value}\n"
            )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print()
    print(
        f"Results saved to: {output_dir.resolve()}"
    )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze local prime distributions using "
            "cyclic residue labeling."
        )
    )

    parser.add_argument(
        "--max-n",
        type=int,
        default=DEFAULT_MAX_N,
        help=(
            "Largest integer to analyze "
            f"(default: {DEFAULT_MAX_N:,})"
        )
    )

    parser.add_argument(
        "--modulus",
        type=int,
        default=DEFAULT_MODULUS,
        help=(
            "Primary modulus "
            f"(default: {DEFAULT_MODULUS})"
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=(
            "Output directory "
            f"(default: {DEFAULT_OUTPUT})"
        )
    )

    args = parser.parse_args()

    if args.max_n < 2:
        raise ValueError(
            "--max-n must be at least 2."
        )

    if args.modulus < 2:
        raise ValueError(
            "--modulus must be at least 2."
        )

    if args.modulus > 26:
        raise ValueError(
            "This implementation supports moduli up to 26 "
            "because labels use A-Z."
        )

    run_experiment(
        max_n=args.max_n,
        modulus=args.modulus,
        output_dir=Path(args.output)
    )


if __name__ == "__main__":
    main()