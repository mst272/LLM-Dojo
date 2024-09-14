import re


def extract_generation_code(example: str, lang_code: str, verbose: bool = False):
    task_id = example['task_id']
    output = example.get('output', example.get("gpt_completion"))
    question = example["prompt"].strip()
    setting = languge_settings[lang_code]
    lang = setting['full_name']
    indent = setting['indent']

    try:
        code_block: str = re.findall(f'```{lang.lower()}\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        if verbose:
            print(">>> Task: {}\n{}".format(task_id, code_block))

        # Remove main
        if setting.get('main', None) and setting['main'] in code_block:
            main_start = code_block.index(setting['main'])
            code_block = code_block[:main_start]

        func_name, func_prefix = get_function_name(question, lang)

        try:
            start = code_block.lower().index(func_name.lower())
            indent = 0
            while start - indent >= 0 and code_block[start - indent - 1] == ' ':
                indent += 1

            try:
                end = code_block.rindex('\n' + ' ' * indent + '}')
            except:
                end = len(code_block)
        except:
            start = 0
            try:
                end = code_block.rindex('\n' + ' ' * indent + '}')
            except:
                end = len(code_block)

        body = code_block[start:end]

        if lang_code.lower() in ['php', 'ts', 'js']:
            body += '\n' + ' ' * indent + '}'

        generation = func_prefix + '\n' + body + '\n'
        example['generation'] = generation

    except Exception as ex:
        print("Failed to extract code block with error `{}`:\n>>> Task: {}\n>>> Output:\n{}".format(
            ex, task_id, output
        ))
        example['generation'] = example['prompt'] + '\n' + output

    return example
