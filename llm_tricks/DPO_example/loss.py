import torch.nn.functional as F
import torch.nn as nn
import torch


# 计算DPO loss的公式
class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
            self,
            policy_chosen_logps: torch.Tensor,
            policy_rejected_logps: torch.Tensor,
            reference_chosen_logps: torch.Tensor,
            reference_rejected_logps: torch.Tensor,
    ):
        """
        policy_chosen_logps: 模型输出的对数概率。Shape: (batch_size,)
        policy_rejected_logps:   Shape: (batch_size,)
        reference_chosen_logps: Shape: (batch_size,)
        reference_rejected_logps: Shape: (batch_size,)
        """
        policy_logps = policy_chosen_logps - policy_rejected_logps
        reference_logps = reference_chosen_logps - reference_rejected_logps
        logits = policy_logps - reference_logps

        loss = -F.logsigmoid(self.beta * logits)

        # 下面两个用于追踪训练的进度
        chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()

        # 对每个batch进行平均(期望)
        return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()


class SimPo(nn.Module):
    """
    SimPO Loss
    """

    def __init__(self, beta: float = 0.1, gamma: float = 0.5) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
            self,
            policy_chosen_logps: torch.Tensor,
            policy_rejected_logps: torch.Tensor,
    ):
        """
        policy_chosen_logps: 模型输出的对数概率。Shape: (batch_size,)
        policy_rejected_logps:   Shape: (batch_size,)
        """
        logits = policy_chosen_logps - policy_rejected_logps
        logits = logits - self.gamma
        loss = -F.logsigmoid(self.beta * logits)

        # 对每个batch进行平均(期望)
        return loss.mean()


# 计算每个模型的Log probabilities
def compute_logprobs(logits, labels, mask=None):
    """
    logits:  shape (batch_size, sequence_len, vocab_size)，即将label输入给模型后输出的结果
    labels:  shape (batch_size, sequence_len)
    """

    # 需要先进行位移操作
    # 去掉标签的第一个
    labels = labels[:, 1:].clone()
    # 去掉模型输出的最后一个
    logits = logits[:, :-1, :]

    logps = F.log_softmax(logits, dim=-1)

    select_logprobs = torch.gather(
        input=logps,
        dim=-1,
        index=labels.unsqueeze(1)
    ).squeeze(1)

    if mask is not None:
        mask = mask[:, 1:].clone()
        # 进行掩码padding部分
        select_logprobs = select_logprobs * mask
        # 计算每一句的平均
        average_logprobs = select_logprobs.sum(-1) / mask.sum(-1)
        return average_logprobs
    else:
        return select_logprobs.mean(-1)


# 计算每个模型的Log probabilities. 使用torch的F.cross_entropy进行计算。结果同上，均是一样。
def compute_logprobs_f_cross(logits, labels, mask=None):
    """
    logits:  shape (batch_size, sequence_len, vocab_size),即将label输入给模型后输出的结果
    labels:  shape (batch_size, sequence_len)
    """
    # 需要先进行位移操作
    # 去掉标签的第一个
    labels = labels[:, 1:].clone()
    # 去掉模型输出的最后一个
    logits = logits[:, :-1, :].clone()

    batch_size, sequence_len, vocab_size = logits.shape
    cross_entropy_loss = 0

    if mask is not None:
        mask = mask[:, 1:].clone()
        labels.masked_fill_(~mask, -100)
        for i in range(batch_size):
            cross_entropy_loss += F.cross_entropy(logits[i], labels[i])
    else:
        for i in range(batch_size):
            cross_entropy_loss += F.cross_entropy(logits[i], labels[i])
    cross_entropy_loss /= batch_size
    return cross_entropy_loss


def compute_batch_loss(batch, policy_model, reference_model, beta):
    # 决定使用哪个loss
    # loss_fn = SimPo(beta, 0.5)   SimPO loss
    loss_fn = DPOLoss(beta)   # DPO loss

    policy_chosen_logps = compute_logprobs(
        logits=policy_model(batch["chosen"]).logits,
        labels=batch["chosen"],
        mask=batch["chosen_mask"]
    )
    policy_rejected_logps = compute_logprobs(
        logits=policy_model(batch["rejected"]).logits,
        labels=batch["rejected"],
        mask=batch["rejected_mask"]
    )
    reference_chosen_logps = compute_logprobs(
        logits=reference_model(batch['chosen']).logits,
        labels=batch['chosen'],
        mask=batch["chosen_mask"]
    )
    reference_rejected_logps = compute_logprobs(
        logits=reference_model(batch['rejected']).logits,
        labels=batch['rejected'],
        mask=batch["rejected_mask"]
    )
    loss, chosen_rewards, rejected_rewards = loss_fn(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
    )
    # SimPO使用如下
    # loss = loss_fn(
    #     policy_chosen_logps=policy_chosen_logps,
    #     policy_rejected_logps=policy_rejected_logps,
    # )
    # return loss
    return loss, chosen_rewards, rejected_rewards


def compute_loss_dataloader(data_loader, policy_model, reference_model, beta, num_batches=5):
    total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
    num_batches = min(num_batches, len(data_loader))

    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss, chosen_rewards, rejected_rewards = compute_batch_loss(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()
        else:
            break
    # 计算平均
    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    return total_loss, total_chosen_rewards, total_rejected_rewards


if __name__ == "__main__":
    # 测试compute_logprobs_f_cross 与 compute_logprobs
    logits = torch.tensor(
        [[2.0, 1.0, 0.1, 0.4],
         [0.5, 2.5, 0.3, 0.5],
         [0.6, 2.5, 0.3, 0.8],
         [0.5, 2.5, 0.6, 0.6]], dtype=torch.float32).unsqueeze(0)
    mask = torch.tensor([[True, True, False, False]])
    targets = torch.tensor([0, 1, 0, 2]).unsqueeze(0)
    loss1 = -compute_logprobs(logits, targets, mask)
    loss2 = compute_logprobs_f_cross(logits, targets, mask)
    print(loss1, loss2)
