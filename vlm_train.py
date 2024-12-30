import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from train_args.vlm_config.script_args import ScriptArgs
from utils.data_process import VlmQaDataset
from utils.data_collator import VlmQaDataCollator


def freeze_vision_projection(model, model_key_value):
    """
    model_key_value: 'projector' or 'visual'
    冻结模型的视觉部分或者转接部分
    """
    module = getattr(model, model_key_value)
    for param in module.parameters():
        param.requires_grad = False


def initial_args():
    parser = TrlParser((ScriptArgs, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False
    return script_args, training_args, model_args


def create_model_processor(model_args, script_args):
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    if script_args.train_mode in ['lora', 'qlora']:
        model_args.target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
        model_args.use_peft = True

        if script_args.train_mode == 'qlora':
            model_args.load_in_4bit = True
            quantization_config = get_quantization_config(model_args)

        model_kwargs = dict(
            revision=model_args.model_revision,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    return {
        'model': model,
        'processor': processor,
        'peft_config': get_peft_config(model_args),
        'target_modules': model_args.target_modules
    }


# 加载数据集，后续不同任务可能会动态调整
def load_vlm_dataset(script_args):
    train_dataset = VlmQaDataset(script_args.train_data_path)
    return train_dataset


def main():
    script_args, training_args, model_args = initial_args()
    train_dict = create_model_processor(model_args, script_args)

    model = train_dict['model']
    processor = train_dict['processor']

    train_dataset = load_vlm_dataset(script_args)
    collate_fn = VlmQaDataCollator(processor)

    model_keys = collate_fn.template_process.model_key
    if script_args.freeze_vision:
        freeze_vision_projection(model, model_keys['visual'])
    if script_args.freeze_projector:
        freeze_vision_projection(model, model_keys['projector'])

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        processing_class=processor.tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()


if __name__ == "__main__":
    main()
