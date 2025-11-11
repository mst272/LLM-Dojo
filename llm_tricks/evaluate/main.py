from transformers import HfArgumentParser
import os
from args import EvaluateArgs
from task import humaneval
from generate import generate_main, evaluation_only

parser = HfArgumentParser((EvaluateArgs,))
args = parser.parse_args_into_dataclasses()[0]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 任务列表
TASKS = {
    "humaneval": humaneval.HumanEval()
}

task = TASKS[args.task_name]

if not args.evaluate_only:
    generate_main(args, task)
else:
    evaluation_only(args, task)
