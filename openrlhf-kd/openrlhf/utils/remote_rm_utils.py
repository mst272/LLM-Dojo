import time
import ray
import requests

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url, data, try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            return response
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


# @ray.remote
# def remote_rm_fn_ray(api_url, queries, prompts, labels):
#     return request_api_wrapper(api_url, {"query": queries, "prompts": prompts, "labels": labels})
@ray.remote
def remote_rm_fn_ray(api_url, queries, prompts, labels, datasources=None,
                     query_token_ids=None, prompt_token_lens=None):
    data = {"query": queries, "prompts": prompts, "labels": labels}
    if datasources is not None:
        data["datasources"] = datasources
    if query_token_ids is not None:
        data["query_token_ids"] = query_token_ids
    if prompt_token_lens is not None:
        data["prompt_token_lens"] = prompt_token_lens
    return request_api_wrapper(api_url, data)


@ray.remote
class RemoteRewardModel:
    def __init__(self, args, remote_rm_url):
        self.args = args
        self.remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        self.custom_reward_func = None

        if self.remote_rm_url and self.remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {self.remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", self.remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = ray.remote(reward_module.reward_func)

    def get_rewards(self, queries_list, prompts_list, labels_list,
                    datasources_list=None, query_token_ids_list=None, prompt_token_lens_list=None):
        if self.custom_reward_func:
            # Let Ray automatically distribute the workload across available resources
            batch_size = self.args.micro_rollout_batch_size
            num_chunks = (len(queries_list) + batch_size - 1) // batch_size
            r_refs = []
            for i in range(num_chunks):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(queries_list))
                
                datasources_chunk = None
                if datasources_list is not None:
                    datasources_chunk = datasources_list[start_idx:end_idx]
                query_token_ids_chunk = None
                if query_token_ids_list is not None:
                    query_token_ids_chunk = query_token_ids_list[start_idx:end_idx]
                prompt_token_lens_chunk = None
                if prompt_token_lens_list is not None:
                    prompt_token_lens_chunk = prompt_token_lens_list[start_idx:end_idx]

                r = self.custom_reward_func.remote(
                    queries_list[start_idx:end_idx],
                    prompts_list[start_idx:end_idx],
                    labels_list[start_idx:end_idx],
                    datasources=datasources_chunk,
                    query_token_ids=query_token_ids_chunk,
                    prompt_token_lens=prompt_token_lens_chunk,
                )
                r_refs.append(r)
        else:
            # Distribute data across different remote reward function servers
            num_servers = len(self.remote_rm_url)
            batch_size = (len(queries_list) + num_servers - 1) // num_servers
            r_refs = []
            for i in range(num_servers):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(queries_list))
                rm = self.remote_rm_url[i]
                
                datasources_chunk = None
                if datasources_list is not None:
                    datasources_chunk = datasources_list[start_idx:end_idx]
                query_token_ids_chunk = None
                if query_token_ids_list is not None:
                    query_token_ids_chunk = query_token_ids_list[start_idx:end_idx]
                prompt_token_lens_chunk = None
                if prompt_token_lens_list is not None:
                    prompt_token_lens_chunk = prompt_token_lens_list[start_idx:end_idx]

                r = remote_rm_fn_ray.remote(
                    rm,
                    queries=queries_list[start_idx:end_idx],
                    prompts=prompts_list[start_idx:end_idx],
                    labels=labels_list[start_idx:end_idx],
                    datasources=datasources_chunk,
                    query_token_ids=query_token_ids_chunk,
                    prompt_token_lens=prompt_token_lens_chunk,
                )
                r_refs.append(r)

        return ray.get(r_refs)
