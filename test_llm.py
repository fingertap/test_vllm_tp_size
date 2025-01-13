import ray

from vllm_v_0_6_3 import LLM


if __name__ == "__main__":
    ray.init()

    model_path = "Qwen/Qwen2.5-72B-Instruct"

    llm1 = (
        ray.remote(LLM)
        .options(
            num_gpus=1,
            runtime_env={
                "env_vars": {
                    "WORLD_SIZE": "2",
                    "RANK": "0",
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": "12355",
                    "RAY_DEDUP_LOGS": "1",
                }
            },
        )
        .remote(model_path, tensor_parallel_size=2)
    )
    llm2 = (
        ray.remote(LLM)
        .options(
            num_gpus=1,
            runtime_env={
                "env_vars": {
                    "WORLD_SIZE": "2",
                    "RANK": "1",
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": "12355",
                    "RAY_DEDUP_LOGS": "1",
                }
            },
        )
        .remote(model_path, tensor_parallel_size=2)
    )

    ray.wait([llm1.init_cache_engine.remote(), llm2.init_cache_engine.remote()])

    futures = [
        llm1.generate.remote(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
            "assistant.<|im_end|>\n<|im_start|>user\nHello, world!<|im_end|>\n"
        ),
        llm2.generate.remote(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
            "assistant.<|im_end|>\n<|im_start|>user\nWho are you?<|im_end|>\n"
        ),
    ]
    print(ray.get(futures))
