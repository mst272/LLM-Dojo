from ..base_utils import TaskUtils
from ..evaluation import evaluate_functional_correctness


class HumanEval(TaskUtils):
    super().__init__()

    @staticmethod
    def build_instruction(example):
        """
        根据模型构建合适的指令
        """
        return example['prompt']

    def generation_code_process(self, example):
        """
        对生成的代码提取函数部分 及 设置import等操作
        """
        code = example['output']
        test_case = example['test']
        code_ = []
        skip_rest = False  # 新增标志位，用于跳过if __name__ == "__main__"及其后面的内容
        for line in code.split("\n"):
            if skip_rest:
                continue  # 如果遇到if __name__ == "__main__"，跳过该行及其后面的所有内容
            if "if __name__ == \"__main__\":" in line:
                skip_rest = True  # 设置标志位，表示需要跳过后续内容
                continue
            if "def " in line and line[0] != ' ' and line[0] != '\t':
                code_.append("def " + line.split("def ")[1])
                continue
            if "class" in line and line.strip().endswith(":"):
                code_.append(line)
                continue
            if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                continue
            code_.append(line)
        code = "\n".join(code_)
        test_setup = "\n".join(self.IMPORT_HELPER["python"]) + "\n"
        example['generation'] = test_setup + code + "\n" + test_case + "\n"
        return example

    @staticmethod
    def evaluate_function(input_file, args):
        """
        最终评测的方法，输入为保存的生成jsonl文件
        """
        return evaluate_functional_correctness(input_file, n_workers=32, timeout=3.0, k=1, save_logs_path=args.save_logs_path)

