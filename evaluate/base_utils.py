class TaskUtils:
    def __init__(self):
        self.IMPORT_HELPER = {
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

    @staticmethod
    def build_instruction(example):
        """
        根据模型构建合适的指令
        """
        return example['prompt']

    @staticmethod
    def generation_code_process(example):
        """
        对生成的代码提取函数部分 及 设置import、添加test用例等操作
        """
        pass

    @staticmethod
    def evaluate_function(input_file, args):
        """
        最终评测的方法，输入为保存的生成jsonl文件
        """
        pass
