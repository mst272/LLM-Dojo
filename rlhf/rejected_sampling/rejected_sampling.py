import asyncio
import aiohttp
from datetime import datetime
import re
from tqdm.auto import tqdm
import string
import json
from typing import List, Dict
import glob
import orjson
import os


def load_completions(input_path: str) -> List[dict]:
    """
    从 JSONL 文件或文件夹中加载 completions 数据 (使用 orjson)。

    Args:
        input_path: 文件路径或文件夹路径。

    Returns:
        包含所有加载的 completions 数据的列表。
    """
    all_completions = []
    if os.path.isfile(input_path):
        with open(input_path, 'rb') as f:  # 注意使用二进制读取模式 'rb'
            for line in f:
                all_completions.append(orjson.loads(line))
    elif os.path.isdir(input_path):
        for filename in glob.glob(os.path.join(input_path, '*.jsonl')):
            print(f"正在加载文件: {filename}")
            with open(filename, 'rb') as f:  # 注意使用二进制读取模式 'rb'
                for line in f:
                    all_completions.append(orjson.loads(line))
    else:
        print(f"错误: 输入路径 '{input_path}' 不是有效的文件或文件夹。")
    return all_completions


def reason_post_process(code, index):
    """
    Args:
        code (str): 输入字符串。
        index (int/str): 当前字符串的序号 (索引)。

    Returns:
        str 或 int: 如果找到代码块，则返回最后一个代码块字符串；
                     否则，返回输入的字符串序号 (index)。
    """

    # Look for code blocks
    code_pattern = r'```(?:python|go|ts|php|csharp|bash|javascript|cpp|cs|java)(.*?)```'
    code_match = re.findall(code_pattern, code, re.DOTALL)

    if code_match:
        # If code block exists, return its content (excluding the ``` markers)
        return code_match[-1].strip()
    else:
        # If no code block, return the solution content directly
        print('---', index)
        # print(code)
        return code


def create_dynamic_comparison_prompt(prompt: str, responses: list[str]) -> str:
    """
    根据给定的 prompt 和一个包含多个代码响应的列表，
    生成一个用于比较这些代码片段的动态提示模板。

    Args:
        prompt: 描述问题的字符串。
        responses: 包含多个代码片段（字符串）的列表。

    Returns:
        一个格式化好的、用于大模型评估的完整提示字符串。
        如果 responses 为空，则返回错误信息。
    """
    if not responses:
        return "错误：未提供任何代码响应进行比较。"

    num_responses = len(responses)

    # 1. 构建模板的静态开头部分
    #    修改引言以适应多个片段
    header = f"""Compare the following {num_responses} code snippets that aim to solve the given problem. 
Evaluate each snippet based on efficiency, readability, and adherence to best practices. 
Identify the preferred snippet or rank them if applicable.

### Problem:
{prompt}
"""

    # 2. 动态构建每个代码片段的部分
    snippets_section = ""
    for i, response in enumerate(responses):
        # 生成标签：Code A, Code B, ...
        if i < 26:  # 最多支持到 Z
            label = string.ascii_uppercase[i]
        else:  # 如果超过 26 个，就用数字编号
            label = str(i + 1)

        snippets_section += f"\n### Code {label}:\n{response}\n"  # 每个代码块前后加换行

    # 3. 构建模板的静态结尾部分
    footer = """
Code Analysis (Provide a brief analysis for each snippet, discussing its pros and cons regarding efficiency, readability, and best practices):

Preferred Code (Output only the single letter label of the most preferred code snippet in 【】 below this line, e.g., 【answer here】):
"""

    # 4. 组合所有部分
    full_prompt = header + snippets_section + footer
    return full_prompt


async def make_request(session: aiohttp.ClientSession, prompt: str, index: int, api_key: str, post_url: str,
                       cookie: str) -> Dict:
    url = post_url
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
        "Cookie": cookie
    }

    payload = {
        "stream": False,
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
        "n": 1
    }

    try:
        async with session.post(url, headers=headers, json=payload, timeout=1000) as response:
            response.raise_for_status()
            json_response = await response.json()
            return {
                'index': index,
                'status': 'success',
                'prompt': prompt,
                'response': reason_post_process(json_response['choices'][0]['message']['content'], index)
            }
    except aiohttp.ClientError as e:
        return {
            'index': index,
            'status': 'error',
            'prompt': prompt,
            'response': f"请求失败：{str(e)}"
        }
    except json.JSONDecodeError as e:
        return {
            'index': index,
            'status': 'error',
            'prompt': prompt,
            'response': f"JSON解析错误：{str(e)}"
        }


def extract_answer(text):
    """
  提取字符串中第一个【】内的内容，并返回字符串。

  Args:
    text: 输入字符串。

  Returns:
    第一个【】内的内容字符串，如果没有任何匹配，则返回 None。
  """
    pattern = r'【(.*?)】'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


###  优化
async def process_group(session: aiohttp.ClientSession, group: dict) -> dict:
    """处理单个分组的异步函数"""
    prompt = group['prompt']
    responses = group['responses']
    full_prompt = create_dynamic_comparison_prompt(prompt, responses)

    # 使用已存在的session发送请求
    result = await make_request(session, full_prompt, 0)  # 索引在这里不重要
    selected_label = extract_answer(result['response'])

    return {
        'prompt': prompt,
        'responses': responses,
        'indices': group['indices'],
        'selected_label': selected_label
    }


async def process_comparisons_async(completions, num_per_group=3, max_concurrent=5):
    start_time = datetime.now()
    grouped_data = []
    current_group = {
        'prompt': None,
        'responses': [],
        'indices': []
    }

    # 1. 分组数据并记录原始索引
    for idx, item in enumerate(completions):
        prompt = item['messages'][0]['content']
        response = item['messages'][1]['content']

        if idx % num_per_group == 0 and idx != 0:
            grouped_data.append(current_group)
            current_group = {
                'prompt': prompt,
                'responses': [response],
                'indices': [idx]
            }
        else:
            if not current_group['responses']:
                current_group['prompt'] = prompt
            current_group['responses'].append(response)
            current_group['indices'].append(idx)

    # 处理最后一组
    if current_group['responses']:
        grouped_data.append(current_group)

    print(f"总共需要处理 {len(grouped_data)} 个分组")

    # 2. 并发处理所有分组
    async with aiohttp.ClientSession() as session:
        tasks = []
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)

        # 将计数器移动到外层
        success_count = 0
        error_count = 0

        # 修改嵌套函数的结构，确保正确使用 nonlocal
        async def process_with_semaphore(group, pbar):
            nonlocal success_count, error_count  # 正确的 nonlocal 声明位置

            async with semaphore:
                try:
                    result = await process_group(session, group)
                    success_count += 1
                    pbar.update(1)
                    pbar.set_postfix({'成功': success_count, '失败': error_count})
                    return result
                except Exception as e:
                    error_count += 1
                    pbar.update(1)
                    pbar.set_postfix({'成功': success_count, '失败': error_count})
                    return {
                        'prompt': group['prompt'],
                        'responses': group['responses'],
                        'indices': group['indices'],
                        'selected_label': None,
                        'error': str(e)
                    }

        with tqdm(total=len(grouped_data), desc="处理进度") as pbar:
            tasks = [process_with_semaphore(group, pbar) for group in grouped_data]
            results = await asyncio.gather(*tasks)

    # 打印统计信息
    end_time = datetime.now()
    print(f"\n处理完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"处理结果统计:")
    print(f"- 成功: {success_count}")
    print(f"- 失败: {error_count}")
    print(f"- 总计: {len(grouped_data)}")
    print(f"- 耗时: {end_time - start_time}")

    # 3. 更新原始数据
    for result in results:
        if result.get('error'):
            # 处理错误情况
            for original_idx in result['indices']:
                completions[original_idx]['comparison'] = {
                    'error': result['error']
                }
            continue

        selected_label = result['selected_label'].strip().upper() if result['selected_label'] else None
        for pos, original_idx in enumerate(result['indices']):
            label = string.ascii_uppercase[pos]
            completions[original_idx]['comparison'] = {
                'group_prompt': result['prompt'],
                'position_label': label,
                'is_best': label == selected_label if selected_label else False,
                'best_label': selected_label,
                'compared_with': len(result['responses'])
            }

    return completions


def _save_chunk_to_jsonl(chunk: List[dict], output_filename: str) -> bool:
    """
    保存一个数据块为 JSONL 格式的文件 (使用 orjson)。

    Args:
        chunk: 要保存的数据块。
        output_filename: 输出文件名。

    Returns:
        True 如果保存成功，False 如果发生错误。
    """
    try:
        with open(output_filename, 'wb') as f:  # 注意使用二进制写入模式 'wb'
            for item in chunk:
                f.write(orjson.dumps(item) + b'\n')  # orjson.dumps 返回 bytes
        return True
    except Exception as e:
        print(f"保存 chunk 到文件 '{output_filename}' 时发生错误: {str(e)}")
        return False


def save_results_in_chunks(completions: List[dict], output_prefix: str = 'output', chunksize: int = 1000):
    """
    将处理结果按 chunksize 分块保存为多个 JSONL 文件 (使用 orjson)。

    Args:
        completions: 处理后的完整数据列表。
        output_prefix: 输出文件名的前缀。
        chunksize: 每个文件保存的数据条数。
    """
    # 提取目录路径
    output_dir = os.path.dirname(output_prefix)
    # 如果目录不存在，则创建目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    num_chunks = (len(completions) + chunksize - 1) // chunksize
    for i in range(num_chunks):
        start_index = i * chunksize
        end_index = min((i + 1) * chunksize, len(completions))
        chunk = completions[start_index:end_index]
        output_filename = f"{output_prefix}_part_{i + 1}.jsonl"
        if _save_chunk_to_jsonl(chunk, output_filename):
            print(f"已保存 chunk {i + 1} 到文件: {output_filename}")


if __name__ == "__main__":
    completions = load_completions('/rejected')
    # completions = completions[:10]
    processed_data = asyncio.run(process_comparisons_async(completions, 3, 60))
    output_file = './v3_eval/v3eval'

    save_results_in_chunks(processed_data, output_file, 16000)
