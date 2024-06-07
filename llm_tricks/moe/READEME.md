# Make MOE step by step

从零构建一个MOE代码存放于 **make_moe_step_by_step.ipynb**文件下。其中有详细的代码注释，推荐结合技术博客阅读，因为博客中手画了许多图以更好地理解。

## 😸技术博客链接

- [从零构建一个MOE](https://zhuanlan.zhihu.com/p/701777558)



## 补充

博客中没提到的一点是 Expert Capacity。大概意思就是为了防止所有tokens都被一个或几个expert处理，我们需要设置一个专家容量。如果某个专家处理超过容量的tokens后就会给他截断，下面给出一个简单的代码示例，实际生产中会有更高级复杂的策略,
例如在https://arxiv.org/abs/2101.03961 中讨论的switch transformer架构。

我们简单的介绍代码如下，与我们技术博客中讲的SparseMoE基本相同，只是加了两个部分，在代码注释中也已标明。
```python
class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))  
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        tokens_per_batch = batch_size * seq_len * self.top_k
        # 定义专家容量
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)

        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            
            # 进行容量判断
            limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                updates.index_add_(0, limited_indices, weighted_output)

        # Reshape updates to match the original dimensions of x
        final_output += updates.view(batch_size, seq_len, -1)

        return final_output

```