import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from execution import check_correctness

IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ]
}


def process_humaneval_test(sample):
    """
    Processes a sample for evaluation.

    return: the str of test code
    """
    test = sample["test"]
    code = sample["generation"]
    test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
    test_string = test_setup + code + "\n" + test + "\n"
    return test_string


def stream_jsonl_all(filename: str):
    """
    Streams a JSONL file.
    """
    results = []
    fp = open(filename, "r")
    for line in fp:
        if any(not x.isspace() for x in line):
            results.append(json.loads(line))
    fp.close()
    return results


def evaluate_functional_correctness(
        input_file: str = None,
        n_workers: int = 32,
        timeout: float = 3.0,
        k: int = 1
):
    """
    Evaluates the functional correctness of a model.
    """
    sample_jsonl = stream_jsonl_all(input_file)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm(sample_jsonl):
            task_id = sample["task_id"]
            sample["test_code"] = process_humaneval_test(sample)
            if sample["test_code"] is None:
                continue
            args = (sample['test_code'], task_id, timeout)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            n_samples += 1

        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append(result)
    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        passed = [r["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()}
    print(pass_at_k)

    return pass_at_k


def estimate_pass_at_k(
        num_samples,
        num_correct,
        k: int
) -> np.ndarray:
    """
    Estimates pass@k and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    assert len(num_samples) == len(num_correct)
    num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
