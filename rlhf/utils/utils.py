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
