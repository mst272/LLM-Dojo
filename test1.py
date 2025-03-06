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
from transformers.deepspeed import is_deepspeed_zero3_enabled
import json


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

        self.trainer = trainer
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.higher_better = higher_better
        self.best_checkpoints: List[CheckpointInfo] = []
        self.checkpoint_dir = os.path.join(trainer.args.output_dir, "best_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 只在主进程中创建meta文件
        if self.trainer.is_world_process_zero():
            self._create_or_load_meta_file()

    def _create_or_load_meta_file(self):
        """创建或加载meta文件来跟踪最佳checkpoints"""
        self.meta_file = os.path.join(self.checkpoint_dir, "best_checkpoints_meta.json")
        if os.path.exists(self.meta_file):
            with open(self.meta_file, 'r') as f:
                meta_data = json.load(f)
                self.best_checkpoints = [
                    CheckpointInfo(**checkpoint_info)
                    for checkpoint_info in meta_data["checkpoints"]
                ]
        else:
            with open(self.meta_file, 'w') as f:
                json.dump({"checkpoints": []}, f)

    def _save_meta_file(self):
        """保存meta信息到文件"""
        if self.trainer.is_world_process_zero():
            with open(self.meta_file, 'w') as f:
                json.dump({
                    "checkpoints": [
                        {
                            "step": cp.step,
                            "metric_value": cp.metric_value,
                            "path": cp.path
                        }
                        for cp in self.best_checkpoints
                    ]
                }, f)

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
        """保存DeepSpeed兼容的checkpoint"""
        checkpoint_folder = f"checkpoint-{step}"
        output_dir = os.path.join(self.checkpoint_dir, checkpoint_folder)

        # 确保所有进程同步
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # 只在主进程中创建目录
        if self.trainer.is_world_process_zero():
            os.makedirs(output_dir, exist_ok=True)

        # 处理DeepSpeed Zero-3的情况
        if is_deepspeed_zero3_enabled():
            # 在Zero-3模式下，需要确保所有进程都参与保存
            self.trainer.model_wrapped.save_checkpoint(output_dir)
        else:
            # 对于其他DeepSpeed配置或非DeepSpeed情况
            success = self.trainer.save_model(output_dir)
            if not success:
                return None

        if self.trainer.is_world_process_zero():
            # 保存tokenizer
            if self.trainer.tokenizer is not None:
                self.trainer.tokenizer.save_pretrained(output_dir)

            # 保存训练参数
            self.trainer.args.save_json(os.path.join(output_dir, "training_args.bin"))

            # 保存优化器状态和调度器
            if self.trainer.deepspeed:
                # DeepSpeed在save_checkpoint时已经保存了优化器状态
                pass
            else:
                self.trainer._save_optimizer_and_scheduler(output_dir)

            # 保存RNG状态
            self.trainer._save_rng_state(output_dir)

            # 保存训练状态
            self.trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # 确保所有进程同步
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return output_dir

    def _update_best_checkpoints(self, state):
        """更新最佳checkpoint列表"""
        current_metric = state.metrics.get(self.metric_name)
        if current_metric is None:
            return

        # 对于越小越好的指标，转换为负数以便统一处理
        metric_value = current_metric if self.higher_better else -current_metric

        # 只在主进程中更新checkpoint列表
        if self.trainer.is_world_process_zero():
            if (len(self.best_checkpoints) < self.max_checkpoints or
                    metric_value > self.best_checkpoints[0].metric_value):

                # 保存新的checkpoint
                checkpoint_path = self._save_checkpoint(state.global_step, metric_value)
                if checkpoint_path is None:
                    return

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

                # 保存meta信息
                self._save_meta_file()

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
        if self.trainer.is_world_process_zero():
            self._wandb.log({"sample_predictions": records_table})

        # 更新最佳checkpoints
        self._update_best_checkpoints(state)