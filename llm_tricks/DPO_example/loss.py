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
        dim=1,
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
