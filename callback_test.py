from dataclasses import dataclass

from transformers.integrations import WandbCallback
import torch
import wandb
from tqdm.auto import tqdm
from transformers import GenerationConfig, Trainer, TrainingArguments


@dataclass
class CheckpointInfo:
    step: int
    metric_value: float
    path: str

    def __lt__(self, other):
        return self.metric_value < other.metric_value


generate_config = GenerationConfig(
    max_new_tokens=512,
    temperature=0.1,
    top_p=1
)


class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = generate_config

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

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions": records_table})


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
