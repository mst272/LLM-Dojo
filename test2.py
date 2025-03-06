import wandb
from transformers.trainer_callback import WandbCallback
from transformers import GenerationConfig
import torch
from tqdm import tqdm
import os
import shutil
from dataclasses import dataclass
from typing import List
import heapq


@dataclass
class CheckpointInfo:
    step: int
    metric_value: float
    path: str

    def __lt__(self, other):
        return self.metric_value < other.metric_value


class LLMSampleCB(WandbCallback):
    def __init__(
            self,
            trainer,
            test_dataset,
            num_samples=10,
            max_new_tokens=256,
            log_model="checkpoint",
            max_checkpoints=3,
            metric_name="eval_loss",
            higher_better=False
    ):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(
            trainer.model.name_or_path,
            max_new_tokens=max_new_tokens
        )

        # 保存trainer引用
        self.trainer = trainer

        # checkpoint相关属性
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.higher_better = higher_better
        self.best_checkpoints: List[CheckpointInfo] = []
        self.checkpoint_dir = os.path.join(trainer.args.output_dir, "best_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
        return records_table

    def _save_checkpoint(self, step, metric_value):
        """保存完整的训练checkpoint"""
        checkpoint_folder = f"checkpoint-{step}"
        output_dir = os.path.join(self.checkpoint_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        # 保存模型
        self.trainer.save_model(output_dir)

        # 保存tokenizer
        if self.trainer.tokenizer is not None:
            self.trainer.tokenizer.save_pretrained(output_dir)

        # 保存优化器和scheduler
        self.trainer._save_optimizer_and_scheduler(output_dir)

        # 保存scaler (用于混合精度训练)
        self.trainer._save_scaler(output_dir)

        # 保存RNG状态
        self.trainer._save_rng_state(output_dir)

        # 保存Trainer状态
        if self.trainer.args.should_save:
            # 更新callback状态
            for cb in [
                cb for cb in self.trainer.callback_handler.callbacks + [self.trainer.control]
                if hasattr(cb, 'state')
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.trainer.state.stateful_callbacks.get(cb_name, None), list):
                    self.trainer.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.trainer.state.stateful_callbacks[cb_name] = cb_state

            # 保存trainer状态
            self.trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        return output_dir

    def _update_best_checkpoints(self, state):
        """更新最佳checkpoint列表"""
        current_metric = state.metrics.get(self.metric_name)
        if current_metric is None:
            return

        # 对于越小越好的指标（如loss），转换为负数以便统一处理
        metric_value = current_metric if self.higher_better else -current_metric

        # 如果还没有达到最大数量，或者当前指标比最差的更好
        if (len(self.best_checkpoints) < self.max_checkpoints or
                metric_value > self.best_checkpoints[0].metric_value):

            # 保存新的checkpoint
            checkpoint_path = self._save_checkpoint(state.global_step, metric_value)

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
                if os.path.exists(worst_checkpoint.path):
                    shutil.rmtree(worst_checkpoint.path)

            # 记录到wandb
            wandb.log({
                "best_checkpoints": [
                    {
                        "step": cp.step,
                        "metric_value": cp.metric_value if self.higher_better else -cp.metric_value,
                        "path": cp.path
                    }
                    for cp in sorted(self.best_checkpoints, key=lambda x: x.metric_value, reverse=True)
                ]
            })

    def on_evaluate(self, args, state, control, **kwargs):
        """评估时的回调函数"""
        super().on_evaluate(args, state, control, **kwargs)

        # 生成样本预测表格
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions": records_table})

        # 更新最佳checkpoints
        self._update_best_checkpoints(state)