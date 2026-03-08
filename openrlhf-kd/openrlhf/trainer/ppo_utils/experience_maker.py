import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import Any, List, Tuple, Union

import ray
import torch
import torch.nn.functional as F

from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.utils import remove_pad_token, zero_pad_sequences

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data for RLHF training.

    Shapes of each tensor:
    index: (B,)
    sequences: (B, S)
    attention_mask: (B, S)
    action_mask: (B, A)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    teacher_log_probs: (B, A)  # teacher model logprobs for KD
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    kl: (B, A)
    info: dict[str, list]
    """

    index: list[int] = None
    sequences: torch.Tensor = None
    attention_mask: torch.LongTensor = None
    action_mask: torch.BoolTensor = None

    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    rollout_log_probs: torch.Tensor = None
    teacher_log_probs: torch.Tensor = None  # teacher model logprobs for knowledge distillation
    values: torch.Tensor = None
    returns: torch.Tensor = None
    advantages: torch.Tensor = None
    kl: torch.Tensor = None

    prompts: list[str] = None
    labels: list[str] = None
    datasources: list[str] = None  # data source for each sample, used for per-datasource reward monitoring
    rewards: torch.Tensor = None  # used for advantage calculation
    scores: torch.Tensor = None  # 0-1 reward used for dynamic sampling

    # the info field is used to store additional information
    # all the fields in the info will be logged to the tensorboard/wandb
    info: dict[str, torch.Tensor] = None

    def __init__(
        self,
        index=None,
        sequences=None,
        action_log_probs=None,
        base_action_log_probs=None,
        rollout_log_probs=None,
        teacher_log_probs=None,
        values=None,
        returns=None,
        advantages=None,
        attention_mask=None,
        action_mask=None,
        kl=None,
        prompts=None,
        labels=None,
        datasources=None,
        rewards=None,
        scores=None,
        info=None,
    ):
        self.index = index
        self.sequences = sequences
        self.action_log_probs = action_log_probs
        self.base_action_log_probs = base_action_log_probs
        self.rollout_log_probs = rollout_log_probs
        self.teacher_log_probs = teacher_log_probs
        self.values = values
        self.returns = returns
        self.advantages = advantages
        self.attention_mask = attention_mask
        self.action_mask = action_mask
        self.kl = kl
        self.prompts = prompts or []
        self.labels = labels or []
        self.datasources = datasources or []
        self.rewards = rewards
        self.scores = scores
        self.info = dict(info) if info is not None else {}

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: to(val, device) for key, val in value.items()})
            else:
                setattr(self, field, to(value, device))

        return self

    def pin_memory(self):
        """Pin memory for all tensor fields."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: pin_memory(val) for key, val in value.items()})
            else:
                setattr(self, field, pin_memory(value))

        return self

    @staticmethod
    def select(experiences: List["Experience"], fields: List[str]) -> List["Experience"]:
        """Select specific fields from a list of Experience instances to create new Experience instances.

        Args:
            experiences: List of Experience instances
            fields: List of field names to select

        Returns:
            A list of new Experience instances containing only the selected fields
        """
        new_experiences = []
        for exp in experiences:
            new_exp = Experience()
            for field in fields:
                if hasattr(exp, field):
                    value = getattr(exp, field)
                    if field == "info" and value is not None:
                        value = dict(value)
                    setattr(new_exp, field, value)
            new_experiences.append(new_exp)
        return new_experiences

    @staticmethod
    def _merge_item(items: List, pad_value: int = 0) -> Union[torch.Tensor, list, dict, Any]:
        """Merge a list of items into a single item.
        Recursively merge tensors, lists and dicts.
        For tensors, use zero_pad_sequences to merge sequences of different lengths.

        Args:
            items: List of items to merge
            pad_value: Value used for padding tensors
        """
        if isinstance(items[0], torch.Tensor):
            return zero_pad_sequences(items, side="right", value=pad_value)
        elif isinstance(items[0], list):
            return sum(items, [])
        elif isinstance(items[0], dict):
            result = {}
            # Collect all values for each key
            for d in items:
                for key, value in d.items():
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            # Merge all values for each key at once
            return {key: Experience._merge_item(values, pad_value) for key, values in result.items()}
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(experiences_list: List["Experience"], pad_token_id) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate
            pad_token_id: Token id used for padding sequences

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()

        # Get all field names from the dataclass
        field_names = [f.name for f in fields(Experience)]

        # Create result dictionary
        result = {}

        # Merge all fields
        for field in field_names:
            values = [getattr(e, field) for e in experiences_list]
            # Use pad_token_id for sequences field, 0 for others
            pad_value = pad_token_id if field == "sequences" else 0
            result[field] = Experience._merge_item(values, pad_value)

        return Experience(**result)


def update_samples_with_rewards(rewards_info, samples_list):
    """Process rewards info and update samples with rewards, scores and extra logs.

    Args:
        rewards_info: List of reward information dictionaries
        samples_list: List of Experience objects to update
    """
    # Process rewards and scores
    samples_len = [len(sample.sequences) for sample in samples_list]

    rewards_list = torch.cat([torch.as_tensor(info["rewards"]) for info in rewards_info], dim=0).split(samples_len)
    if "scores" in rewards_info[0]:
        scores_list = torch.cat([torch.as_tensor(info["scores"]) for info in rewards_info], dim=0).split(samples_len)
    else:
        scores_list = rewards_list

    # Process extra_logs if present
    # TODO: keep only keys shared by all batches, then concatenate and split.
    if "extra_logs" in rewards_info[0]:
        # Find keys that exist in ALL extra_logs (intersection)
        all_extra_logs = [info["extra_logs"] for info in rewards_info]
        common_keys = set(all_extra_logs[0].keys())
        for logs in all_extra_logs[1:]:
            common_keys &= set(logs.keys())
        
        # Merge only common extra_logs tensors
        merged_logs = {
            key: torch.cat([logs[key] for logs in all_extra_logs], dim=0).split(samples_len)
            for key in common_keys
        }

    # Handle teacher_log_probs for knowledge distillation
    # teacher_log_probs from reward function is a list of per-sample 1D tensors (variable length, no padding)
    has_teacher_log_probs = "teacher_log_probs" in rewards_info[0]
    if has_teacher_log_probs:
        all_teacher_log_probs = sum([info["teacher_log_probs"] for info in rewards_info], [])

    # Update samples with rewards, scores and extra logs
    offset = 0
    for i, samples in enumerate(samples_list):
        samples.rewards = rewards_list[i]
        samples.scores = scores_list[i]
        samples.info["score"] = scores_list[i]
        samples.info["reward"] = rewards_list[i]
        if "extra_logs" in rewards_info[0]:
            for key, values in merged_logs.items():
                samples.info[key] = values[i]

        # Assign teacher_log_probs: pad each per-sample tensor to match action_mask length
        if has_teacher_log_probs:
            batch_size = len(samples.sequences)
            target_len = samples.action_mask.size(1)  # (B, A) -> A
            sample_tlps = all_teacher_log_probs[offset : offset + batch_size]
            teacher_lens = torch.tensor([len(tlp) for tlp in sample_tlps], dtype=torch.float32)
            padded_tlps = torch.stack(
                [
                    F.pad(torch.as_tensor(tlp).float(), (0, max(0, target_len - len(tlp))), value=0.0)[:target_len]
                    for tlp in sample_tlps
                ]
            )
            samples.teacher_log_probs = padded_tlps

            # Teacher response diagnostics for KD observability.
            valid_tokens = samples.action_mask.float().sum(dim=-1).clamp(min=1)
            covered_tokens = torch.minimum(teacher_lens, valid_tokens)
            samples.info["teacher_cov_ratio"] = covered_tokens / valid_tokens
            samples.info["teacher_len"] = teacher_lens
            samples.info["teacher_finite_ratio"] = torch.isfinite(samples.teacher_log_probs).float().mean(dim=-1)

            offset += batch_size

    return samples_list


class SamplesGenerator:
    def __init__(self, vllm_engines, strategy, tokenizer, prompt_max_len):
        self.strategy = strategy
        self.args = strategy.args
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len

    @torch.no_grad()
    #def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Experience]:
    def generate_samples(
        self, all_prompts: List[str], all_labels, all_datasources: List[str] = None, **generate_kwargs
    ) -> List[Experience]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        #rollout_samples = self._generate_vllm(all_prompts, all_labels, **generate_kwargs)
        rollout_samples = self._generate_vllm(all_prompts, all_labels, all_datasources, **generate_kwargs)


        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        return rollout_samples

        # tokenizer

    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

   # def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Experience]:
    def _generate_vllm(
        self, all_prompts: List[str], all_labels, all_datasources: List[str] = None, **kwargs
    ) -> List[Experience]:
        """Generate samples using vLLM engine.

        Args:
            all_prompts: List of prompts to generate from
            all_labels: List of labels corresponding to prompts
            all_datasources: List of data source identifiers corresponding to prompts
            **kwargs: Additional arguments for generation

        Returns:
            List of Experience objects containing generated samples
        """
        from vllm import SamplingParams

        llms = self.vllm_engines
        args = self.strategy.args

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
            logprobs=1 if self.strategy.args.enable_vllm_is_correction else None,
        )
        max_response_length = kwargs.get("max_new_tokens", 1024)
        truncate_length = self.prompt_max_len + max_response_length

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)
        all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        
        # Expand datasources list if provided
        if all_datasources is not None:
            all_datasources = sum([[ds] * n_samples_per_prompt for ds in all_datasources], [])
        else:
            all_datasources = ["default"] * len(all_prompts)

        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(llm.add_requests.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))
        ray.get(refs)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_outputs = sum(ray.get(all_output_refs), [])

        # Process outputs into Experience objects
        samples_list = []
        for i in range(len(all_outputs)):
            output = all_outputs[i]
            prompt = all_prompts[i]
            label = all_labels[i]
            datasource = all_datasources[i]

            # Concatenate prompt and output tokens
            input_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
            attention_mask = [1] * len(input_ids)

            sequences = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

            # Create action mask based on output token positions
            action_mask = torch.zeros_like(attention_mask)
            response_length = len(output.outputs[0].token_ids)
            action_mask[len(output.prompt_token_ids) : len(output.prompt_token_ids) + response_length] = 1

            # Calculate rollout log probs
            rollout_log_probs = None
            if self.strategy.args.enable_vllm_is_correction:
                rollout_log_probs = []
                response_ids = list(output.outputs[0].token_ids)
                for i, logprob in enumerate(output.outputs[0].logprobs):
                    rollout_log_probs.append(logprob[response_ids[i]].logprob)

                rollout_log_probs = torch.tensor([0.0] * len(list(output.prompt_token_ids)) + rollout_log_probs)
                rollout_log_probs = rollout_log_probs[1:truncate_length].to("cpu")

            sequences = sequences[:truncate_length].to("cpu")
            attention_mask = attention_mask[:truncate_length].to("cpu")
            action_mask = action_mask[1:truncate_length].to("cpu")
            total_length = attention_mask.float().sum()
            is_clipped = response_length >= max_response_length

            prompt_token_len = len(output.prompt_token_ids)

            info = {
                "response_length": torch.tensor([response_length]),
                "total_length": torch.tensor([total_length]),
                "response_clip_ratio": torch.tensor([is_clipped]),
                "prompt_token_len": torch.tensor([prompt_token_len]),
            }

            rollout_samples = Experience(
                sequences=sequences.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                action_mask=action_mask.unsqueeze(0),
                rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
                prompts=[prompt],
                labels=[label],
                datasources=[datasource],
                info=info,
            )
            samples_list.append(rollout_samples)

        # Get rewards from remote reward models if needed
        # This is required by dynamic sampling
        remote_reward_model = kwargs.get("remote_reward_model", None)
        if remote_reward_model:
            all_queries = sum(
                [
                    self.tokenizer.batch_decode(
                        remove_pad_token(s.sequences, s.attention_mask), skip_special_tokens=False
                    )
                    for s in samples_list
                ],
                [],
            )
            all_query_token_ids = None
            all_prompt_token_lens = None
            _needs_kd = (
                self.strategy.args.advantage_estimator == "kd"
                or getattr(self.strategy.args, "kd_coef", 0) > 0
            )
            if _needs_kd:
                all_query_token_ids = sum(
                    [[ids.tolist() for ids in remove_pad_token(s.sequences, s.attention_mask)] for s in samples_list],
                    [],
                )
                all_prompt_token_lens = sum(
                    [s.info["prompt_token_len"].tolist() for s in samples_list],
                    [],
                )
            all_prompts = sum([s.prompts for s in samples_list], [])
            all_labels = sum([s.labels for s in samples_list], [])
            all_datasources = sum([s.datasources for s in samples_list], [])


            # Get rewards info from remote model (with datasources for routing)
            rewards_info = ray.get(remote_reward_model.get_rewards.remote(
                all_queries, all_prompts, all_labels, all_datasources,
                all_query_token_ids, all_prompt_token_lens,
            ))

            # Process rewards and scores
            update_samples_with_rewards(rewards_info, samples_list)

        return samples_list


class RemoteExperienceMaker(ABC):
    def __init__(
        self,
        actor_model_group: RayActorGroup,
        initial_model_group: RayActorGroup,
        kl_controller,
        strategy=None,
        tokenizer=None,
        remote_reward_model=None,
        **kwargs,
    ):
        super().__init__()

        self.actor_model_group = actor_model_group
        self.initial_model_group = initial_model_group
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.advantage_estimator = strategy.args.advantage_estimator
        self.args = strategy.args

        # remote_rm_url indicates that the remote reward model is agent enviroment, remote http server or custom reward func
        self.remote_rm_url = self.args.remote_rm_url
        self.remote_reward_model = remote_reward_model
        self.tokenizer = tokenizer

    def split_rollout_samples(self, rollout_samples):
        '''
        Split rollout samples into smaller batches for efficient processing.
        This function handles both dynamic batching (when use_dynamic_batch is True)
        and fixed batching (when use_dynamic_batch is False).
        and will padding the sequences to the same length for each batch.
        '''
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list = []
        if self.args.use_dynamic_batch:
            total_lengths = [int(s.info["total_length"].item()) for s in rollout_samples]
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )
            minimum_batch_num = get_minimum_num_micro_batch_size(
                total_lengths,
                self.args.rollout_max_tokens_per_gpu,
                self.args.ring_attn_size,
                self.args.ds_tensor_parallel_size,
            )
            minimum_batch_num = minimum_batch_num // effective_actor_num * effective_actor_num
            num_batch = max(minimum_batch_num, effective_actor_num)
            batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)
            for micro_index in batch_indexes:
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(micro_batch, self.tokenizer.pad_token_id)
                samples_list.append(concat_samples)
        else:
            batch_size = self.args.micro_rollout_batch_size
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = Experience.concat_experiences(
                    rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)
        return samples_list

    @torch.no_grad()
    def make_experience_batch(self, rollout_samples) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        samples_list = self.split_rollout_samples(rollout_samples)

        # Make experiences (models forward: logprobs, rewards, and kl divergence)
        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences

    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, rewards, and kl divergence.
        """
        start_time = time.time()
        logger.info(f"🚀 Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples")

        args = self.strategy.args
        device = "cpu"

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]

        # The rewards are already filled in the samples_list, such as the agent's environment rewards
        if samples_list[0].rewards is not None:
            pass
        elif self.remote_rm_url:
            queries_list = sum(
                [
                    self.tokenizer.batch_decode(remove_pad_token(seq, attention_mask), skip_special_tokens=False)
                    for seq, attention_mask in zip(sequences_list, attention_mask_list)
                ],
                [],
            )
            query_token_ids_list = None
            prompt_token_lens_list = None
            _needs_kd = (
                self.advantage_estimator == "kd"
                or getattr(self.args, "kd_coef", 0) > 0
            )
            if _needs_kd:
                query_token_ids_list = sum(
                    [
                        [ids.tolist() for ids in remove_pad_token(seq, attention_mask)]
                        for seq, attention_mask in zip(sequences_list, attention_mask_list)
                    ],
                    [],
                )
                prompt_token_lens_list = sum(
                    [s.info["prompt_token_len"].tolist() for s in samples_list],
                    [],
                )
            prompts_list = sum([s.prompts for s in samples_list], [])
            labels_list = sum([s.labels for s in samples_list], [])
            
            datasources_list = sum([s.datasources for s in samples_list], [])
            r_refs = self.remote_reward_model.get_rewards.remote(
                queries_list, prompts_list, labels_list, datasources_list,
                query_token_ids_list, prompt_token_lens_list,
            )
        else:
            raise ValueError("Local reward model is disabled. Please configure --remote_rm_url or --agent_func_path.")

        # Batch call actor model
        action_log_probs_ref = self.actor_model_group.async_run_method_batch(
            method_name="forward",
            sequences=sequences_list,
            action_mask=action_mask_list,
            attention_mask=attention_mask_list,
        )

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models or args.colocate_actor_ref:
            ray.get(action_log_probs_ref)
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

        # Batch call initial model
        if self.initial_model_group is not None:
            base_action_log_probs_ref = self.initial_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )

            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(base_action_log_probs_ref)
                ray.get(self.initial_model_group.async_run_method(method_name="empty_cache"))
        else:
            base_action_log_probs_ref = ray.put(
                [[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size)
            )

        # Wait for all remote calls to complete and flatten the results
        # Note: the results duplicated ring_attn_size * ds_tensor_parallel_size times
        # This is because the actors in ring group and tp group will return the same output
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        action_log_probs_list = sum(ray.get(action_log_probs_ref)[::duplicate_factor], [])
        base_action_log_probs_list = sum(ray.get(base_action_log_probs_ref)[::duplicate_factor], [])

        # Process rewards based on source
        if samples_list[0].rewards is not None:
            pass
        elif self.remote_rm_url:
            # Get rewards info from remote model
            rewards_info = ray.get(r_refs)
            # Process rewards and scores
            update_samples_with_rewards(rewards_info, samples_list)

        assert len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list), (
            f"len(samples_list): {len(samples_list)}, "
            f"len(action_log_probs_list): {len(action_log_probs_list)}, "
            f"len(base_action_log_probs_list): {len(base_action_log_probs_list)}"
        )

        # Process results for each sample
        for i, (samples, action_log_probs, base_action_log_probs) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list)
        ):
            if (self.initial_model_group is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
            kl_mean = masked_mean(kl, samples.action_mask, dim=-1)

            if not args.use_kl_loss:
                base_action_log_probs = None

            # Update experience with new information
            samples.action_log_probs = action_log_probs
            samples.base_action_log_probs = base_action_log_probs
            samples.kl = kl
            samples.info["kl"] = kl_mean

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str}")
        return samples_list

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.
        Example, use_dynamic_batch
            >>> rewards: [0, 1, 0.5, 1], indices: [1, 2, 0, 3], n_samples_per_prompt: 2
            >>> sorted rewards: [0,5, 0, 1, 1], reward shaping: [0.25, 0.25, 1, 1]
            >>> map back: [0.25, 1, 0.25, 1]
        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args

        # Knowledge Distillation mode: advantage = teacher_log_probs - action_log_probs
        if self.advantage_estimator == "kd":
            return self._compute_kd_advantages(experiences)

        # DAPO reward shaping with optional overlong penalty - Apply BEFORE dynamic indices processing
        if args.overlong_buffer_len is not None:
            assert (
                args.generate_max_len >= args.overlong_buffer_len
            ), "generate_max_len must be larger than overlong_buffer_len"
            overlong_buffer_len = args.overlong_buffer_len
            expected_len = args.generate_max_len - overlong_buffer_len
            overlong_penalty_factor = args.overlong_penalty_factor

            # Apply penalty to each experience's rewards based on response length
            for experience in experiences:
                response_lengths = experience.info["response_length"]
                batch_size = len(response_lengths)
                for j in range(batch_size):
                    valid_response_length = response_lengths[j].item()
                    # Cap the exceed_len to overlong_buffer_len to prevent excessive penalty
                    exceed_len = min(valid_response_length - expected_len, overlong_buffer_len)
                    if exceed_len > 0:
                        overlong_penalty = -exceed_len / overlong_buffer_len * overlong_penalty_factor
                        # Apply penalty to the j-th reward in this experience
                        experience.rewards[j] += overlong_penalty

        # get rewards from experiences
        exp_len = [len(experience.index) for experience in experiences]
        # indices is an identity mapping when not using dynamic batch; otherwise, it maps back to the original indices after rearange samples
        indices = torch.tensor(sum([experience.index for experience in experiences], []))
        raw_rewards = torch.cat([experience.rewards for experience in experiences], dim=0)
        rewards = torch.empty_like(raw_rewards)
        rewards[indices] = raw_rewards  # sorted

        rewards = rewards.reshape(-1, args.n_samples_per_prompt)

        # log group reward std
        if args.n_samples_per_prompt > 1:
            group_reward_stds = (
                rewards.std(-1, keepdim=True).repeat(1, args.n_samples_per_prompt).reshape(-1)[indices].split(exp_len)
            )
            for experience, group_reward_std in zip(experiences, group_reward_stds):
                experience.info["group_reward_std"] = group_reward_std

        # reward shaping
        if args.advantage_estimator == "rloo":
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = rewards - rewards.mean(-1, keepdim=True)
        elif args.advantage_estimator == "group_norm":
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)

        rewards = rewards.reshape(-1)[indices].split(exp_len)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                if args.gamma != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    args.gamma = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    args.gamma,
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            return_sums = reward.sum(dim=-1)
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
        
        # --------- Hybrid KD + RLVR overlay ---------
        kd_coef = getattr(args, "kd_coef", 0.0)
        if kd_coef > 0:
            self._overlay_kd_advantages(experiences, kd_coef)

        # Normalize advantages across all experiences
        if self.args.advantage_estimator in ["reinforce", "reinforce_baseline"]:
            all_advantages = []
            all_action_masks = []
            for exp in experiences:
                all_advantages.append(exp.advantages.flatten())
                all_action_masks.append(exp.action_mask.flatten())

            advantages_vector = torch.cat(all_advantages, dim=0).float()
            action_masks_vector = torch.cat(all_action_masks, dim=0)
            num_actions = action_masks_vector.sum()

            # mean
            mean = (advantages_vector * action_masks_vector).sum() / num_actions
            # std
            if not self.args.no_advantage_std_norm:
                var = ((advantages_vector - mean).pow(2) * action_masks_vector).sum() / num_actions
                rstd = var.clamp(min=1e-8).rsqrt()
            else:
                rstd = 1

            # Apply normalization to each experience
            for exp in experiences:
                exp.advantages = (exp.advantages - mean) * rstd

        return experiences

    @torch.no_grad()
    def _overlay_kd_advantages(self, experiences: List[Experience], kd_coef: float):
        """
        Overlay KD advantages onto existing RLVR advantages for hybrid KD+RLVR mode.

        For samples where kd_skip_reward == 1 (reward was skipped, pure KD):
            advantage = kd_advantage  (ignore rlvr_advantage entirely)
        For samples where kd_mask == 1 and kd_skip_reward == 0 (KD+RLVR blend):
            advantage = (1 - kd_coef) * rlvr_advantage + kd_coef * kd_advantage
        For samples where kd_mask == 0:
            advantage remains the original rlvr_advantage (unchanged).
        """
        kd_sample_count = 0
        kd_skip_count = 0
        for experience in experiences:
            kd_mask = experience.info.get("kd_mask", None)
            if kd_mask is None or experience.teacher_log_probs is None:
                continue

            has_kd = kd_mask > 0.5  # (B,)
            if not has_kd.any():
                continue

            kd_advantages = (experience.teacher_log_probs - experience.action_log_probs).float()
            kd_advantages = kd_advantages * experience.action_mask

            # Determine per-sample blending coefficient:
            # - kd_skip_reward samples → coef = 1.0 (pure KD, reward was skipped)
            # - normal KD samples      → coef = kd_coef (standard blend)
            kd_skip = experience.info.get("kd_skip_reward", None)
            if kd_skip is not None:
                is_skip = (kd_skip > 0.5).float()  # (B,)
                per_sample_coef = is_skip + (1.0 - is_skip) * kd_coef  # (B,)
                kd_skip_count += (has_kd & (kd_skip > 0.5)).sum().item()
            else:
                per_sample_coef = torch.full_like(kd_mask, kd_coef)

            # Blend: mask and coef broadcast from (B, 1) to (B, A)
            mask = has_kd.float().unsqueeze(-1)  # (B, 1)
            coef = per_sample_coef.unsqueeze(-1)  # (B, 1)
            experience.advantages = (
                experience.advantages * (1.0 - coef * mask)
                + coef * kd_advantages * mask
            )
            experience.returns = experience.advantages.clone()

            # Diagnostics
            action_mask_f = experience.action_mask.float()
            token_count = action_mask_f.sum(dim=-1).clamp(min=1.0)
            teacher_mean = masked_mean(experience.teacher_log_probs, experience.action_mask, dim=-1)
            student_mean = masked_mean(experience.action_log_probs, experience.action_mask, dim=-1)
            kd_adv_mean = (kd_advantages * action_mask_f).sum(dim=-1) / token_count
            experience.info["hybrid_kd_teacher_logp_mean"] = teacher_mean
            experience.info["hybrid_kd_student_logp_mean"] = student_mean
            experience.info["hybrid_kd_adv_mean"] = kd_adv_mean

            kd_sample_count += has_kd.sum().item()

        if kd_sample_count > 0:
            total = sum(len(exp.index) for exp in experiences)
            msg = f"Hybrid KD+RLVR: {kd_sample_count}/{total} samples received KD overlay (kd_coef={kd_coef})"
            if kd_skip_count > 0:
                msg += f", {kd_skip_count} samples used pure KD (kd_skip_reward)"
            logger.info(msg)


    @torch.no_grad()
    def _compute_kd_advantages(self, experiences: List[Experience]) -> List[Experience]:
        """
        Compute advantages for knowledge distillation mode.
        advantage = teacher_log_probs - action_log_probs (token-level)
        
        Unlike standard RL which uses scalar rewards, KD directly computes token-level advantages
        from teacher and student log probabilities. No reward shaping, GAE, or cumulative returns needed.
        """
        args = self.strategy.args

        for experience in experiences:
            assert experience.teacher_log_probs is not None, (
                "teacher_log_probs is required for KD advantage estimator. "
                "Make sure your reward function returns 'teacher_log_probs'."
            )

            # advantage = teacher_log_probs - student_log_probs (token-level)
            advantages = (experience.teacher_log_probs - experience.action_log_probs).float()
            # Apply action_mask: only response tokens contribute, prompt and padding are zeroed
            advantages = advantages * experience.action_mask

            action_mask_f = experience.action_mask.float()
            token_count = action_mask_f.sum(dim=-1).clamp(min=1.0)
            finite_mask = torch.isfinite(experience.teacher_log_probs) & torch.isfinite(experience.action_log_probs)
            finite_ratio = (finite_mask.float() * action_mask_f).sum(dim=-1) / token_count
            teacher_mean = masked_mean(experience.teacher_log_probs, experience.action_mask, dim=-1)
            student_mean = masked_mean(experience.action_log_probs, experience.action_mask, dim=-1)
            kd_abs_mean = masked_mean(advantages.abs(), experience.action_mask, dim=-1)
            adv_mean = (advantages * action_mask_f).sum(dim=-1) / token_count
            adv_centered = advantages - adv_mean.unsqueeze(-1)
            kd_var = ((adv_centered.pow(2)) * action_mask_f).sum(dim=-1) / token_count
            kd_std = torch.sqrt(kd_var + 1e-8)

            experience.advantages = advantages
            experience.returns = advantages.clone()

            # Log return info (sum of per-token advantages)
            return_sums = advantages.sum(dim=-1)
            experience.info["return"] = return_sums
            experience.info["kd_teacher_logp_mean"] = teacher_mean
            experience.info["kd_student_logp_mean"] = student_mean
            experience.info["kd_adv_abs_mean"] = kd_abs_mean
            experience.info["kd_adv_std"] = kd_std
            experience.info["kd_finite_ratio"] = finite_ratio

            teacher_cov_ratio = experience.info.get("teacher_cov_ratio", None)
            if teacher_cov_ratio is not None and torch.any(teacher_cov_ratio < 0.9):
                logger.warning(
                    "KD teacher coverage is low in this batch: min_teacher_cov_ratio=%.4f",
                    teacher_cov_ratio.min().item(),
                )
            if torch.any(finite_ratio < 1.0):
                logger.warning(
                    "KD detected non-finite log probs: min_kd_finite_ratio=%.4f",
                    finite_ratio.min().item(),
                )
            # Remove unnecessary info
            experience.kl = None

        # Normalize advantages across all experiences
        all_advantages = []
        all_action_masks = []
        for exp in experiences:
            all_advantages.append(exp.advantages.flatten())
            all_action_masks.append(exp.action_mask.flatten())

        advantages_vector = torch.cat(all_advantages, dim=0).float()
        action_masks_vector = torch.cat(all_action_masks, dim=0)
        num_actions = action_masks_vector.sum()

        # mean normalization
        mean = (advantages_vector * action_masks_vector).sum() / num_actions
        # std normalization (optional, controlled by no_advantage_std_norm)
        if not getattr(args, "no_advantage_std_norm", False):
            var = ((advantages_vector - mean).pow(2) * action_masks_vector).sum() / num_actions
            rstd = var.clamp(min=1e-8).rsqrt()
        else:
            rstd = 1

        # Apply normalization to each experience
        for exp in experiences:
            exp.advantages = (exp.advantages - mean) * rstd

        return experiences

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns
