# Openrlhf 参数指南

max_samples：使用的最多数据量，假设有1w条数据，但max_samples设置为1k，那么实际只使用1k数据。

rollout_batch_size：推理生成时的prompt数量，与n_samples_per_prompt相乘得到总的推理数量。 wandb上的step就是数据量除这个rollout_batch_size。

n_samples_per_prompt：对每个prompt进行几次生成

micro_rollout_batch_size：每个GPU推理生成的数量

train_batch_size：每次训练更新的数量

micro_train_batch_size：每个GPU训练的数量



