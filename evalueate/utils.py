import re


def extract_generation_code(example, lang_code):
    task_id = example['task_id']
    output = example.get('output')
    prompt = example["prompt"].strip()
    setting = language_settings[lang_code]
    lang = setting['full_name']

    try:
        code_block: str = re.findall(f'```{lang.lower()}\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        # assert code_block.startswith(prompt)
        generation = code_block[len(prompt):]
        example['generation'] = generation

    except Exception as ex:
        print("Failed to extract code block with error `{}`:\n>>> Task: {}\n>>> Output:\n{}".format(
            ex, task_id, output
        ))
        example['generation'] = example['prompt'] + '\n' + output

    return example


language_settings = {
    'python': {
        'full_name': 'Python'
    },
    'cpp': {
        'full_name': 'cpp'
    },
    'java': {
        'full_name': 'Java'
    }
}


class TaskUtils:
    @staticmethod
    def build_instruction(prompt: str):
        """
        根据模型构建合适的指令
        """
        return prompt

    @staticmethod
    def generation_process(code, test):
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
        test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
        return test_setup + code











