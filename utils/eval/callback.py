import heapq
import os
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import List, Optional

import torch
import deepspeed
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
import pandas as pd
from tqdm.auto import tqdm
from trl.import_utils import is_deepspeed_available, is_rich_available, is_vllm_available
from eval_utils import _generate_completions, reason_post_process, pad
import wandb
from datasets import Dataset
from transformers.trainer_callback import ExportableState
from transformers import (
    GenerationConfig,
    Trainer,
    TrainerCallback
)
from eval_accuracy import CodeEvalMetric


@dataclass
class CheckpointInfo:
    step: int
    metric_value: float
    path: str

    def __lt__(self, other):
        return self.metric_value < other.metric_value


class EvaluationCallback(TrainerCallback):
    r"""
    A [`~transformers.TrainerCallback`] that logs completions and eval metrics to Weights & Biases and/or Comet.

    Usage:
    ```python
    trainer = Trainer(...)
    evaluation_callback = EvaluationCallback(trainer=trainer)
    trainer.add_callback(evaluation_callback)
    ```

    Args:
        trainer (`Trainer`):
            Trainer to which the callback will be attached. The trainer's evaluation dataset must include a `"prompt"`
            column containing the prompts for generating completions.
        generation_config (`GenerationConfig`, *optional*):
            The generation config to use for generating completions.
        num_samples (`int` or `None`, *optional*):
            The number of prompts to eval for. If not provided, defaults to the number of examples in the evaluation dataset.
        freq (`int` or `None`, *optional*):
            The frequency at which to log completions. If not provided, defaults to the trainer's `eval_steps`.
        metric
    """

    def __init__(
            self,
            trainer: Trainer,
            test_dataset: Dataset,
            generation_config: GenerationConfig,
            num_samples: Optional[int] = None,
            freq: Optional[int] = None,
            metric: Optional[CodeEvalMetric] = None,
            max_checkpoints: int = 3,
            per_device_test_batch_size: int = 1,
            higher_better: bool = True,
            start_update_best_checkpoints: int = 0,
            gather_deepspeed3_params: bool = True,
            use_vllm: bool = True

    ):
        self.gen_config = generation_config
        self.trainer = trainer
        self.best_checkpoints: List[CheckpointInfo] = []
        self.max_checkpoints = max_checkpoints  # 最大保存数量
        self.higher_better = higher_better  # 指标是否越大越好
        self._last_logged_step = -1
        self.batch_size = per_device_test_batch_size
        self.table = []
        self.freq = freq
        self.metric = metric
        self.start_update_best_checkpoints = start_update_best_checkpoints
        self.gather_deepspeed3_params = gather_deepspeed3_params

        if self.metric is None:
            raise ValueError("You must provide a metric[CodeEvalMetric]")

        if num_samples is not None:
            self.sample_dataset = test_dataset.select(range(num_samples))
        else:
            self.sample_dataset = test_dataset

        # 配置vllm client
        if use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
        if self.trainer.accelerator.is_main_process:
            self.vllm_client = VLLMClient()
        self._last_loaded_step = 0
        self.trainer.accelerator.wait_for_everyone()

    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
        deepspeed_plugin = self.trainer.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext

        if is_peft_model(self.trainer.model):
            # With PEFT and DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as merging
            # adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.trainer.model.parameters())):
                self.trainer.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                for name, param in self.trainer.model.named_parameters():
                    # When using PEFT, we need to recover the original parameter name and discard some parameters
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if self.trainer.model.prefix in name:
                        continue
                    # When module to save, remove its prefix and discard the original module
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")

                    if self.trainer.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

                # Unmerge adapters while parameters are still gathered
                self.trainer.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather and update each parameter individually.
            for name, param in self.trainer.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.trainer.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

        # Reset cache on main process
        if self.trainer.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()

    def samples_generate_split_between_processes(self, examples, steps):
        """
        if model very large, maybe OOM
        """
        labels = [example['message'][1]['content'] for example in self.sample_dataset]
        tokenizer = self.trainer.processing_class
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped
        start_time = time.time()
        with accelerator.split_between_processes(self.sample_dataset['message']) as prompts_split:
            prompts = []
            for lis in prompts_split:
                prompts.append('补全下面代码，将最终题目和答案返回在代码框中\n' + lis[0]['content'])
            completions = _generate_completions(
                prompts,
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                generation_config=self.gen_config,
                batch_size=self.batch_size,
                gather_deepspeed3_params=self.gather_deepspeed3_params
            )
            completions = gather_object(completions)
            prompts = gather_object(prompts)
        end_time = time.time()  # 记录 _generate_completions 结束时间
        generation_time = end_time - start_time  # 计算生成耗时

        generations = [[reason_post_process(c, i)] for i, c in enumerate(completions)]
        print(f"Process {accelerator.process_index}: Generation time: {generation_time:.4f} seconds")

        if len(self.sample_dataset) < accelerator.num_processes:
            generations = generations[:len(labels)]
        # 处理输出表格数据
        if self.trainer.accelerator.is_main_process:
            global_step = [str(steps)] * len(prompts)
            config_keys = list(self.gen_config.to_dict().keys())
            config_values = list(self.gen_config.to_dict().values())
            data = [[global_step[i], prompts[i], completions[i]] + config_values for i in range(len(prompts))]
            self.table.extend(data)
            table = pd.DataFrame(columns=["step", "prompt", "completion"] + config_keys, data=self.table)
            wandb.log({"completions": table})

        score = self.metric.compute(references=labels, predictions=generations)
        return score

    def samples_generate_vllm(self, examples, steps):
        device = self.trainer.accelerator.device
        labels = [example['message'][1]['content'] for example in examples]
        prompts_split = examples['message']
        prompts = []
        for lis in prompts_split:
            prompts.append('补全下面代码，将最终题目和答案返回在代码框中\n' + lis[0]['content'])
        prompts = ['补全下面代码，将最终题目和答案返回在代码框中\n' + prompt for prompt in prompts]

        # First, have main process load weights if needed
        if steps != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = steps
        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_prompts_text = gather_object(prompts)
        if self.trainer.accelerator.is_main_process:
            # todo: with profiling_context(self, "vLLM.generate"):  上下文时间处理
            start_time = time.time()
            completion_ids = self.vllm_client.generate(
                prompts=all_prompts_text,
                max_tokens=4096
            )
            end_time = time.time()  # 记录 _generate_completions 结束时间
            generation_time = end_time - start_time  # 计算生成耗时
            print(f"Process main: Generation time: {generation_time:.4f} seconds")
        else:
            completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        process_slice = slice(
            self.trainer.accelerator.process_index * len(prompts),
            (self.trainer.accelerator.process_index + 1) * len(prompts),
        )
        completion_ids = completion_ids[process_slice]

        tokenizer = self.trainer.processing_class
        tokenizer.padding_side = "left"
        completions = tokenizer.batch_decode(completion_ids)
        generations = [[reason_post_process(c, i)] for i, c in enumerate(completions)]
        # 处理输出表格数据
        if self.trainer.accelerator.is_main_process:
            global_step = [str(steps)] * len(prompts)
            data = [[global_step[i], prompts[i], completions[i]] for i in range(len(prompts))]
            self.table.extend(data)
            table = pd.DataFrame(columns=["step", "prompt", "completion"], data=self.table)
            wandb.log({"completions": table})

        score = self.metric.compute(references=labels, predictions=generations)
        return score

    def samples_generate(self, examples, steps):
        # records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        labels = [example['message'][1]['content'] for example in examples]

        prompts_split = examples['message']
        prompts = []
        for lis in prompts_split:
            prompts.append('补全下面代码，将最终题目和答案返回在代码框中\n' + lis[0]['content'])
        prompts = ['补全下面代码，将最终题目和答案返回在代码框中\n' + prompt for prompt in prompts]

        tokenizer = self.trainer.processing_class
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped
        start_time = time.time()
        completions = _generate_completions(
            prompts,
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            generation_config=self.gen_config,
            batch_size=self.batch_size,
            gather_deepspeed3_params=self.gather_deepspeed3_params
        )
        end_time = time.time()  # 记录 _generate_completions 结束时间
        generation_time = end_time - start_time  # 计算生成耗时

        generations = [[reason_post_process(c, i)] for i, c in enumerate(completions)]
        print(f"Process {accelerator.process_index}: Generation time: {generation_time:.4f} seconds")

        # 处理输出表格数据
        if self.trainer.accelerator.is_main_process:
            # global_step = [str(steps)] * len(prompts) config = list(self.gen_config.to_dict().values()) * len(
            # prompts) data = list(zip(global_step, prompts, completions, config)) self.table.extend(data) table =
            # pd.DataFrame(columns=["step", "prompt", "completion"] + list(self.gen_config.to_dict().keys()),
            # data=self.table) wandb.log({"completions": table})
            global_step = [str(steps)] * len(prompts)
            config_keys = list(self.gen_config.to_dict().keys())
            config_values = list(self.gen_config.to_dict().values())
            data = [[global_step[i], prompts[i], completions[i]] + config_values for i in range(len(prompts))]
            self.table.extend(data)
            table = pd.DataFrame(columns=["step", "prompt", "completion"] + config_keys, data=self.table)
            wandb.log({"completions": table})

        score = self.metric.compute(references=labels, predictions=generations)
        return score

    def save_best_metric_model(self, args, state):
        # Save model checkpoint
        print('开始保存checkpoint')
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = os.path.join(args.output_dir, 'best_model', checkpoint_folder)

        self.trainer.save_model(output_dir)

        if not args.save_only_model:
            # Save optimizer and scheduler
            self.trainer._save_optimizer_and_scheduler(output_dir)
            self.trainer._save_scaler(output_dir)
            # Save RNG state
            self.trainer._save_rng_state(output_dir)
        if args.should_save:
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
        if state.global_step < self.start_update_best_checkpoints:
            return
        print('更新最佳checkpoint列表')
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

    def on_step_end(self, args, state, control, **kwargs):
        # Only log once per step (this method may be called multiple times)
        if state.global_step == self._last_logged_step:
            return

        # Only log every `freq` steps
        freq = self.freq
        if state.global_step % freq != 0:
            return

        # custom_score = self.samples_generate(self.sample_dataset, state.global_step)
        custom_score = self.samples_generate_vllm(self.sample_dataset, state.global_step)
        self.trainer.log({"custom_score": custom_score, "step": state.global_step})

        self.update_best_checkpoints(args, state, custom_score)
        # Save the last logged step, so we don't log the same completions multiple times
        self._last_logged_step = state.global_step
