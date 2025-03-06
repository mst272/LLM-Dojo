import heapq
import shutil
from dataclasses import dataclass
from typing import List

from transformers.integrations import WandbCallback
import torch
import wandb
import os
from tqdm.auto import tqdm
from transformers import GenerationConfig, Trainer, TrainingArguments
from transformers.trainer_callback import ExportableState
import evaluate


@dataclass
class CheckpointInfo:
    step: int
    metric_value: float
    path: str

    def __lt__(self, other):
        return self.metric_value < other.metric_value


def compute_metrics(generations, labels):
    metric = evaluate.load('./metrics/code_eval')
    pass_at_k, results = metric.compute(
        predictions=generations,
        references=labels,
        k=[1]  # 虽然这里指定 k=[1]，但 code_eval 通常默认会计算 pass@1，可以省略 k 参数，
    )


class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        super().__init__()
        generate_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=1
        )
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
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
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt)
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
        score = 1
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
        self._wandb.log({"sample_predictions": records_table})
        self._wandb.log({"custom_score": custom_score, "step": state.global_step})


batch_size = 16
gradient_accumulation_steps = 2
num_train_epochs = 3

training_args = TrainingArguments(
    output_dir="./output/",
    report_to="wandb",  # this tells the Trainer to log the metrics to W&B
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size // 2,
    bf16=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    num_train_epochs=num_train_epochs,
    # logging strategies
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="epoch",  # saving is done at the end of each epoch
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

wandb_callback = LLMSampleCB(trainer, test_dataset, num_samples=10, max_new_tokens=256)
trainer.add_callback(wandb_callback)

trainer.train()
