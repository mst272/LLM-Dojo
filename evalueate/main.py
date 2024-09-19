from transformers import HfArgumentParser
import os
from args import EvaluateArgs
from task import humaneval

parser = HfArgumentParser((EvaluateArgs,))
args = parser.parse_args_into_dataclasses()[0]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TASKS = {
    "humaneval": humaneval.HumanEval
}

task = TASKS[args.task_name]

if not args.generate_only:
    generate_main(args)