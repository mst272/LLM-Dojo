import heapq
import shutil
from dataclasses import dataclass
from random import random
from typing import List, Optional
import re

from accelerate.utils import gather_object
from transformers.integrations import WandbCallback
import torch
from datasets import load_dataset
import wandb
import os
from trl.models.utils import unwrap_model_for_generation
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import GenerationConfig, Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_callback import ExportableState
import evaluate
from utils import MultiRoundDataProcess, SftDataCollator
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# deepspeed需要==0.15.0
@dataclass
class CheckpointInfo:
    step: int
    metric_value: float
    path: str

    def __lt__(self, other):
        return self.metric_value < other.metric_value


def compute_metrics(generations, labels):
    """
    generations: [[str][str]...]
    labels: [str,str,...]
    """
    metric = evaluate.load('./metrics/code_eval')
    pass_at_k, results = metric.compute(
        predictions=generations,
        references=labels,
        k=[1]  # 虽然这里指定 k=[1]，但 code_eval 通常默认会计算 pass@1，可以省略 k 参数，
    )
    return pass_at_k


def reason_post_process(code, index):
    """

    Args:
        code (str): 输入字符串。
        index (int/str): 当前字符串的序号 (索引)。

    Returns:
        str 或 int: 如果找到代码块，则返回代码块字符串；
                     否则，返回输入的字符串序号 (index)。
    """

    # Look for code blocks
    code_pattern = r'```(?:python|go|javascript|java)(.*?)```'
    code_match = re.findall(code_pattern, code, re.DOTALL)

    if code_match:
        # If code block exists, return its content (excluding the ``` markers)
        return code_match[-1].strip()
    else:
        # If no code block, return the solution content directly
        return str(index)


def _generate_completions(
        prompts: list[str],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator,
        generation_config: Optional[GenerationConfig],
        batch_size: int = 1,
) -> list[str]:
    """
    Generates completions for a list of pre-formatted prompts from the given model.

    Args:
        prompts (list[str]): A list of input prompts for which completions are to be generated.
        model (PreTrainedModel): The pre-trained model to be used for generation.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to be used for encoding and decoding.
        accelerator (Accelerator): The accelerator to be used for model execution.
        generation_config (GenerationConfig): Configuration for text generation.
        batch_size (int, optional): The number of prompts to process in each batch. Default is 1.

    Returns:
        list[str]: A list of generated text completions corresponding to the input prompts.
    """
    completions = []
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        for idx in range(0, len(prompts), batch_size):
            batch = prompts[idx: idx + batch_size]
            tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            generations = unwrapped_model.generate(
                **tokenized_batch,
                generation_config=generation_config,
            )
            for prompt, generation in zip(tokenized_batch.input_ids, generations):
                # Remove prompt from generation
                generation = generation[len(prompt):]
                completion = tokenizer.decode(generation, skip_special_tokens=True)
                completions.append(completion)
    return completions


class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        super().__init__()
        generate_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            max_length=512
        )
        # self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model_wrapped, trainer.processing_class
        self.gen_config = generate_config
        self.trainer = trainer
        self.best_checkpoints: List[CheckpointInfo] = []
        self.max_checkpoints = 3  # 最大保存数量
        self.higher_better = True  # 指标是否越大越好

    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(inputs=tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

    def samples_table(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        generations = []
        labels = []
        for example in tqdm(examples, leave=False):
            prompt = '补全下面代码，将最终题目和答案返回在代码框中\n' + example['message'][0]['content']
            label = example['message'][1]['content']
            generation = self.generate(prompt=prompt)
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
            generation = reason_post_process(generation, 1)
            generations.append([generation])
            labels.append(label)
        score = compute_metrics(labels=labels, generations=generations)
        return records_table, score

    def samples_table1(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        # self.tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        labels = [example['message'][1]['content'] for example in examples]
        model = self.trainer.model_wrapped
        with accelerator.split_between_processes(examples['message']) as prompts:
            q = []
            for lis in prompts:
                q.append('补全下面代码，将最终题目和答案返回在代码框中\n' + lis[0]['content'])
            # prompts = ['补全下面代码，将最终题目和答案返回在代码框中\n' + prompt for prompt in prompts]
            prompts = q
            completions = _generate_completions(
                prompts,
                model=model,
                tokenizer=self.tokenizer,
                accelerator=accelerator,
                generation_config=self.gen_config,
                batch_size=1,
                # batch_size=args.per_device_eval_batch_size,
            )
            completions = gather_object(completions)
            prompts = gather_object(prompts)
        generations = [[c] for c in completions]
        score = compute_metrics(labels=labels, generations=generations)
        return records_table, score

    def save_best_metric_model(self, args, state):
        # Save model checkpoint
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)

        self.trainer.save_model(output_dir, _internal_call=True)

        if not args.save_only_model:
            # Save optimizer and scheduler
            self.trainer._save_optimizer_and_scheduler(output_dir)
            self.trainer._save_scaler(output_dir)
            # Save RNG state
            self.trainer._save_rng_state(output_dir)

        # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
        for cb in [
            cb for cb in self.trainer.callback_handler.callbacks + [self.trainer.control] if
            isinstance(cb, ExportableState)
        ]:
            cb_name = cb.__class__.__name__
            cb_state = cb.state()
            if isinstance(state.stateful_callbacks[cb_name], list):
                state.stateful_callbacks[cb_name].append(cb_state)
            else:
                state.stateful_callbacks[cb_name] = cb_state
        state.save_to_json(os.path.join(output_dir, 'trainer_state.json'))

        return output_dir

    def update_best_checkpoints(self, args, state, custom_score):
        """更新最佳checkpoint列表"""

        # 对于越小越好的指标（如loss），转换为负数以便统一处理
        metric_value = custom_score if self.higher_better else -custom_score

        # 如果还没有达到最大数量，或者当前指标比最差的更好
        if (len(self.best_checkpoints) < self.max_checkpoints or
                metric_value > self.best_checkpoints[0].metric_value):

            # 保存新的checkpoint
            checkpoint_path = self.save_best_metric_model(args, state)

            # 创建新的CheckpointInfo对象
            checkpoint_info = CheckpointInfo(
                step=state.global_step,
                metric_value=metric_value,
                path=checkpoint_path
            )

            # 更新最佳checkpoint列表
            heapq.heappush(self.best_checkpoints, checkpoint_info)

            # 如果超过最大数量，删除最差的checkpoint
            if len(self.best_checkpoints) > self.max_checkpoints:
                worst_checkpoint = heapq.heappop(self.best_checkpoints)
                print(f"Deleting older checkpoint [{worst_checkpoint.path}] due to args.save_total_limit")
                shutil.rmtree(worst_checkpoint.path, ignore_errors=True)

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        records_table, custom_score = self.samples_table(self.sample_dataset)
        if self.trainer.is_world_process_zero():
            # self.trainer.log({"sample_predictions": records_table})  # trainer中打印table会有bug
            # self.trainer.log({"custom_score": custom_score, "step": state.global_step})
            self._wandb.log({"sample_predictions": records_table})
            self._wandb.log({"custom_score": custom_score, "step": state.global_step})
            self.update_best_checkpoints(args, state, custom_score)


batch_size = 2
gradient_accumulation_steps = 2
num_train_epochs = 2

training_args = TrainingArguments(
    output_dir="./output/",
    report_to="wandb",  # this tells the Trainer to log the metrics to W&B
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=10,
    bf16=True,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=2,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    eval_strategy="steps",
    num_train_epochs=num_train_epochs,
    # logging strategies
    logging_strategy="steps",
    logging_steps=1,
    eval_steps=3,
    remove_unused_columns=False,
    deepspeed='ds_config_zero3.json'
)

if __name__ == "__main__":
    model_name_or_path = 'Qwen2___5-Coder-7B-Instruct'
    train_data_path = 'y_train.jsonl'
    eval_data_path = '/eval_1.jsonl'
    test_data_path = 'test.jsonl'
    max_len = 1024
    auto_adapt = False

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

    train_dataset = MultiRoundDataProcess(train_data_path, tokenizer, max_len, auto_adapt)

    test_dataset = load_dataset(path="json", data_files=test_data_path)
    test_dataset = test_dataset['train']

    eval_dataset = MultiRoundDataProcess(eval_data_path, tokenizer, max_len, auto_adapt)
    data_collator = SftDataCollator(tokenizer, max_len)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    # if os.environ.get('LOCAL_RANK', '0') == '0':  # 只在主进程中初始化
    #     wandb.init(project="huggingface")
    # wandb.init(project="huggingface")

    wandb_callback = LLMSampleCB(trainer, test_dataset, num_samples=4, max_new_tokens=256)
    trainer.add_callback(wandb_callback)

    trainer.train()
