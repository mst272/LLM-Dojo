from ..base_utils import TaskUtils


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
        example['generation'] = test_setup + code
        return test_setup + code

    @staticmethod
    def return_test_code(sample):
        """
        返回最终进行运行测试code
        """
        test = sample["test"]
        code = sample["generation"]
        test_string = code + "\n" + test + "\n"
        return test_string
