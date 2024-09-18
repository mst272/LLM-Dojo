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
