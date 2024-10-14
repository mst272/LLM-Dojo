from datasets import load_dataset
import random
from transformers import AutoTokenizer
from trl import (
    GKDConfig,
    GKDTrainer,
    LogCompletionsCallback,
    ModelConfig,
    SFTScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from accelerate import PartialState


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, GKDConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    teacher_model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset(data_files=args.dataset_name, path='json')    # 适配jsonl格式

    with PartialState().local_main_process_first():
        dataset = dataset.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(x["prompt"], tokenize=False, add_generation_prompt=True)
            },
            num_proc=training_args.dataset_num_proc,
        )
    train_data = dataset['train']
    test_data = train_data.select(random.sample(range(len(train_data)), 20))

    ################
    # Training
    ################
    trainer = GKDTrainer(
        model=model_config.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=test_data,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )
    completions_callback = LogCompletionsCallback(trainer, trainer.generation_config, num_prompts=8)
    trainer.add_callback(completions_callback)
    trainer.train()

    # Save
    trainer.save_model(training_args.output_dir)
