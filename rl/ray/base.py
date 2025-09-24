import os
import socket
import ray
import torch
from deepspeed.deepspeed import DeepspeedStrategy

# 处理分布式环境设置,设置环境变量、通信参数
class BaseDistributedActor:
    def __init__(self, world_size: int, rank: int, master_addr: str, master_port: str):
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ['MASTER_ADDR'] = self._master_addr
        os.environ['MASTER_PORT'] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)

        os.environ['LOCAL_RANK'] = str(ray.get_gpu_ids()[0]) if self._ray_noset_visible_devices() else '0'
    

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        return address.strip('[]')

    
    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(('', 0))
            return sock.getsockname()[1]

    @staticmethod
    def _ray_noset_visible_devices(env_vars=os.environ):
        NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
        return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)

    def get_master_addr_port(self):
        return self._master_addr, self._master_port



# 模型actor的基类、actor的分布式设置
class BaseModelActor(BaseDistributedActor):
    def _setup_distributed(self, strategy: DeepspeedStrategy):
        self.strategy = strategy
        strategy.setup_distributed() 

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def execute_batch(self):
        pass




# 奖励模型
@ray.remote(num_gpus=1)   # 另一种写法是：RewardModelActor = ray.remote(num_gpus=1)(RewardModelActor)
class ReferenceModelActor(BaseModelActor):
    def init_model_from_pretrained(self, stategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(stategy)
        model = Actor()
        