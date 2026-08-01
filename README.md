# Yahya-s-Cyclic-Prime-Residue-Framework

# Cyclic Prime Residue Analysis

> **Metadata:** Authored on March 3, 2025, during the author's fourteenth year.

A Python-based computational framework designed to investigate the local distribution of prime numbers under a periodic residue-class labeling system.

---

## Overview

This program divides integers into consecutive blocks based on a chosen modulus (e.g., $m = 7$) and maps them to a repeating cyclic alphabetical labeling sequence (where $1 \rightarrow A, 2 \rightarrow B, \dots, 7 \rightarrow G$). It measures local clustering, residue frequencies, and structural shifts to analyze short-range patterns in prime distribution.

> *Important:* This program investigates empirical patterns. It does not assume that any observed pattern is statistically significant or mathematically new.
> 
> 

---

## Key Features

* **High-Performance Sieve:** Utilizes an optimized Sieve of Eratosthenes with NumPy arrays for rapid prime generation up to large integer limits.


* **Block & Residue Analysis:** Automatically segments integers into consecutive blocks to evaluate local prime counts, specific residue occurrences, and global frequency distributions.


* **Local Deviation Tracking:** Calculates how individual block distributions deviate from expected global baselines.


* **Transition Analysis:** Tracks and quantifies directional shifts and structural transitions between neighboring local patterns.


* **Automated Data Export & Visualization:** Generates publication-quality plots via `matplotlib` and exports complete metrics into structured CSV datasets and text summaries.



---

## Command-Line Interface (CLI) Usage

Run the analysis directly from the terminal with customizable arguments for integer limits, moduli, and output directories:

```bash
python cyclic_prime_analysis.py --max-n 1000000 --modulus 7 --output prime_residue_results

```

### Arguments

* `--max-n`: The largest integer to analyze (default: `1,000,000`).


* `--modulus`: The primary modulus for cyclic labeling, supporting values up to 26 corresponding to alphabetic labels A through Z (default: `7`).


* `--output`: Directory path where all exported CSV files, text summaries, and high-resolution PNG figures will be saved (default: `prime_residue_results`).



---

## Dependencies

* Python 3.8+
* NumPy


* Pandas


* Matplotlib
