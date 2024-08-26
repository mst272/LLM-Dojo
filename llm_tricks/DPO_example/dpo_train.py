from torch.utils.data import DataLoader, random_split
import torch
from dataset import RlhfDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from loss import compute_batch_loss
from evaluate import evaluate_loss_dataloader
import time
from functools import partial

# 1、加载模型与tokenizer
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model_path = '/IndexTeam/Index-1___9B-Chat'
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
ref_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
ref_model.eval()
model.to(device)
ref_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

# 2、处理数据
# 加载数据
data_file = './unsloth_dpo.jsonl'
# Dataset详细逻辑可看进入RlhfDataset实现
dataset = RlhfDataset(data_file, tokenizer)
# 划分训练集验证集
train_size = int(len(dataset) * 0.85)  # 85% for training
val_size = len(dataset) - train_size  # Remaining for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 编写batch批次的padding及mask处理函数
IGNORE_INDEX = False


def data_collate(batch, pad_token_id, device, max_length=None, if_mask_prompt=True):
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": []
    }

    # 判断长度及padding
    max_length_common = 0
    for key in ["chosen", "rejected"]:
        current_max = max(len(item[key]) for item in batch)
        max_length_common = max(max_length_common, current_max)

    # 转为torch tensor并padding,决定是否对prompt进行mask
    for item in batch:
        prompt = torch.tensor(item['prompt'])
        batch_data['prompt'].append(prompt)

        for key in ["chosen", "rejected"]:
            out = item[key]
            out_padding = out + [pad_token_id] * (max_length_common - len(out))
            mask = torch.ones(len(out_padding)).bool()

            # padding部分的mask设置为 IGNORE_INDEX
            mask[len(out):] = IGNORE_INDEX

            if if_mask_prompt:
                mask[:prompt.shape[0] + 2] = IGNORE_INDEX
            batch_data[key].append(torch.tensor(out_padding))
            batch_data[f"{key}_mask"].append(mask)

    # 进行最大长度截断
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        tensor_stack = torch.stack(batch_data[key])
        if max_length is not None:
            tensor_stack = tensor_stack[:, :max_length]
        # 将tensor移到对应的device
        batch_data[key] = tensor_stack.to(device)
    return batch_data


customized_collate_fn = partial(
    data_collate,
    pad_token_id=tokenizer.pad_token_id,
    device=device,
    if_mask_prompt=True,
    max_length=1024
)
# 设置相关参数
batch_size = 4
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False
)


# 3、开始计算DPO(或其他)的损失函数
# 相关代码可以再loss里查看，就不写在主函数里了。

# 4、编写训练函数
def train_model(
        policy_model, reference_model, train_loader, val_loader,
        optimizer, num_epochs, beta,
        eval_freq, eval_iter):
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": []
    }
    tokens_seen, global_step = 0, -1

    # 训练
    for epoch in range(num_epochs):
        # policy 模型需要训练
        policy_model.train()

        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss, chosen_rewards, rejected_rewards = compute_batch_loss(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            tokens_seen += batch["chosen"].numel()

            # 验证
            if global_step % eval_freq == 0:
                res = evaluate_loss_dataloader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    beta=beta,
                    eval_iter=eval_iter
                )
                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)
                train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )

    return tracking


# 5、开始训练！
def main():
    torch.manual_seed(42)
    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    num_epochs = 3
    tracking = train_model(
        policy_model=model,
        reference_model=ref_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        beta=0.1,  # value between 0.1 and 0.5
        eval_freq=2,
        eval_iter=2
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")


if __name__ == "__main__":
    main()
