import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from tqdm import tqdm

from .execution import check_correctness

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


def stream_jsonl_all(filename: str) -> Iterable[Dict]:
    """
    Streams a JSONL file.
    """
    results = []
    if filename.endswith(".gz"):
        fp = gzip.open(open(filename, "rb"), "rt")
    else:
        fp = open(filename, "r")
    for line in fp:
        if any(not x.isspace() for x in line):
            results.append(json.loads(line))
    fp.close()

    return results


def evaluate_functional_correctness(
        input_file: str = None,
        tmp_dir: str = "./",
        n_workers: int = 32,
        timeout: int = 3
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
            tmp_dir_ = os.path.join(tmp_dir, "evaluation")
            sample["task_id"] = task_id
            sample["test_code"] = process_humaneval_test(sample)
            if sample["test_code"] is None:
                continue
            args = (sample['test_code'],timeout)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            n_samples += 1

        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()

    return pass_at_k


def evaluate_score(args):
    gs, (c, i, o), mode = args

    execution_results = []
    for g in gs:
        code_to_execute = f"{c}\nassert {o} == {g}"
        execution_results.append(check_correctness(code_to_execute, 3))
    if True not in execution_results:
        execution_results = [False] * len(gs)
    return execution_results
