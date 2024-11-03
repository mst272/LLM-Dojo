from typing import List, Dict
import copy


def is_right_apply_chat(tokenizer, prompt: List[Dict[str, str]], assistant_content: List[Dict[str, str]]) -> bool:
    """
    Checks if the assistant's content is correctly applied to the prompt in a chat template.
    Args:
        tokenizer: The tokenizer.
        prompt: The initial prompt message.
        assistant_content: The content provided by the assistant.
    Returns:
        bool: True if the assistant's content is correctly applied, False otherwise.
    """
    try:
        test_assistant = tokenizer.apply_chat_template(assistant_content, tokenize=False)
        test_prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
        conversation = copy.deepcopy(prompt)
        conversation.append(assistant_content[0])
        if tokenizer.apply_chat_template(conversation) == test_prompt + test_assistant:
            return True
        else:
            return False
    except Exception as e:
        return False


def fix_chat_template_if_needed(tokenizer, prompt: List[Dict[str, str]], chosen: List[Dict[str, str]],
                                rejected: List[Dict[str, str]]):
    """
    Fixes the chat template if needed.
    Args:
        tokenizer: The tokenizer.
        prompt: The initial prompt message.
        chosen: The chosen response, a list containing a single dictionary representing the chosen message.
        rejected: The rejected response, a list containing a single dictionary representing the rejected message.
        Returns:
        - tuple: A tuple containing the fixed prompt, fixed chosen response, and fixed rejected response.
    """
    conversation_chosen = copy.deepcopy(prompt)
    conversation_rejected = copy.deepcopy(prompt)
    conversation_chosen.append(chosen[0])
    conversation_rejected.append(rejected[0])
    conversation_chosen = tokenizer.apply_chat_template(conversation_chosen, tokenize=False)
    conversation_rejected = tokenizer.apply_chat_template(conversation_rejected, tokenize=False)
    # find position
    start_position = conversation_chosen.find(chosen[0]['content'][0])
    # The following is right
    fixed_prompt = conversation_chosen[:start_position]
    fixed_chosen = conversation_chosen[start_position:]
    fixed_rejected = conversation_rejected[start_position:]
    return fixed_prompt, fixed_chosen, fixed_rejected


# def split_data(raw_datasets, eval_samples):
#     train_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples))
#     eval_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples, len(raw_datasets)))
#     return train_dataset, eval_dataset


# def load_data_prompt(tokenizer, train_data_path, eval_samples):
#     raw_datasets = pd.read_json(train_data_path, lines=True)
#     for i in range(len(raw_datasets)):
#         pro = raw_datasets['prompt'][i]
#         res = tokenizer.apply_chat_template(pro, tokenize=False)
#         raw_datasets.loc[i, 'prompt'] = res
#     raw_datasets = Dataset.from_pandas(raw_datasets, preserve_index=False)
#
#     def tokenize(element):
#         outputs = tokenizer(
#             element['prompt'],
#             padding=False,
#         )
#         return {"input_ids": outputs["input_ids"]}
#
#     raw_datasets = raw_datasets.map(
#         tokenize,
#         remove_columns=raw_datasets.column_names,
#         batched=True,
#         num_proc=multiprocessing.cpu_count(),  # multiprocessing.cpu_count(),
#         load_from_cache_file=False,
#     )
#     train_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples))
#     eval_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples, len(raw_datasets)))
#     return train_dataset, eval_dataset


# def test_tokenizer_chat_template(tokenizer, prompt, chosen):
#     test_prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
#     test_chosen = tokenizer.apply_chat_template(chosen, tokenize=False)
#     one_conversation = copy.deepcopy(prompt)
#     one_conversation.append(chosen[-1])
#     one_conversation = tokenizer.apply_chat_template(one_conversation, tokenize=False)
#     if one_conversation == test_prompt + test_chosen:
#         return True
#     else:
#         logger.warning("Chat template is not right, Start automatic repair!")
#         return False
#

# def load_data_all(tokenizer, train_data_path, eval_samples):
#     raw_datasets = pd.read_json(train_data_path, lines=True)
#     prompt, chosen = raw_datasets['prompt'][0], raw_datasets['chosen'][0]
#     if not test_tokenizer_chat_template(tokenizer, prompt, chosen):
#         for i in range(len(raw_datasets)):
#             conversation_chosen = raw_datasets['prompt'][i][:]
#             conversation_rejected = raw_datasets['prompt'][i][:]
#             conversation_chosen.append(raw_datasets['chosen'][i][-1])
#             conversation_rejected.append(raw_datasets['rejected'][i][-1])
#             conversation_chosen = tokenizer.apply_chat_template(conversation_chosen, tokenize=False)
#             conversation_rejected = tokenizer.apply_chat_template(conversation_rejected, tokenize=False)
#             start_position = conversation_chosen.find(raw_datasets['chosen'][i][-1]['content'])
#             raw_datasets.loc[i, 'prompt'] = conversation_chosen[:start_position]
#             raw_datasets.loc[i, 'chosen'] = conversation_chosen[start_position:]
#             raw_datasets.loc[i, 'rejected'] = conversation_rejected[start_position:]
#     else:
#         for i in range(len(raw_datasets)):
#             raw_datasets.loc[i, 'prompt'] = tokenizer.apply_chat_template(raw_datasets['prompt'][i], tokenize=False)
#             raw_datasets.loc[i, 'chosen'] = raw_datasets.loc[i, 'prompt'] + tokenizer.apply_chat_template(
#                 raw_datasets['chosen'][i], tokenize=False)
#             raw_datasets.loc[i, 'rejected'] = raw_datasets.loc[i, 'prompt'] + tokenizer.apply_chat_template(
#                 raw_datasets['rejected'][i], tokenize=False)
#     raw_datasets = Dataset.from_pandas(raw_datasets, preserve_index=False)
#     logger.warning("Now, the chat template is not right!")
#     train_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples))
#     eval_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples, len(raw_datasets)))
#     return train_dataset, eval_dataset