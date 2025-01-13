from __future__ import annotations

import os

from vllm.executor.executor_base import ExecutorBase
from vllm.model_executor.layers.sampler import SamplerOutput


class SPMDGPUExecutor(ExecutorBase):
    def _init_executor(self) -> None:
        assert not self.speculative_config, "Speculative decoding not yet supported for multi-GPU backend."

        # Create the parallel worker for each GPU.
        self._init_workers_sp("env://")

    def _init_workers_sp(self, distributed_init_method: str):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from .worker import Worker

        rank = int(os.getenv("RANK", "0"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        # see https://github.com/NVIDIA/nccl/issues/1234
        os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self.worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            self.cache_config,
            self.load_config,
            local_rank,
            rank,
            distributed_init_method,
            lora_config=self.lora_config,
            speculative_config=None,
            prompt_adapter_config=self.speculative_config,
            is_driver_worker=True,
            model_runner_cls=None,  # use the default one
        )

        self.worker.init_device()
        self.worker.load_model()

    def cpu(self):
        self.worker.cpu()

    def meta(self):
        self.worker.meta()

    def cuda(self):
        self.worker.cuda()

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.worker.update_weight(name, dtype, shape, empty_cache)

    def determine_num_available_blocks(self) -> tuple[int, int]:
        """Determine the number of available KV blocks.

        This invokes `determine_num_available_blocks` on each worker and takes
        the min of the results, guaranteeing that the selected cache sizes are
        compatible with all workers.

        Returns:
            - tuple[num_gpu_blocks, num_cpu_blocks]
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        return self.worker.determine_num_available_blocks()

    def execute_model(self, execute_model_req) -> list[SamplerOutput]:
        return self.worker.execute_model(execute_model_req=execute_model_req)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Initialize the KV cache in all workers."""

        # NOTE: We log here to avoid multiple logs when number of workers is
        # greater than one. We could log in the engine, but not all executors
        # have GPUs.
        from loguru import logger

        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks, num_cpu_blocks)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_cache_engine(self) -> None:
        self.worker._init_cache_engine()

    def free_cache_engine(self) -> None:
        self.worker.free_cache_engine()

    def list_loras(self) -> set[int]:
        return self.worker.list_loras()

    def list_prompt_adapters(self) -> set[int]:
        return self.worker.list_prompt_adapters()

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.worker.pin_lora(lora_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, "prompt_adapter_id must be greater than 0."
        return self.worker.pin_prompt_adapter(prompt_adapter_id)

    def add_lora(self, lora_request) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self.worker.add_lora(lora_request=lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.worker.remove_lora(lora_id=lora_id)

    def add_prompt_adapter(self, prompt_adapter_request) -> bool:
        assert prompt_adapter_request.prompt_adapter_id > 0, "prompt_adapter_id must be greater than 0."
        return self.worker.add_prompt_adapter(prompt_adapter_request=prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, "prompt_adapter_id must be greater than 0."
        return self.worker.remove_prompt_adapter(prompt_adapter_id=prompt_adapter_id)

    def check_health(self) -> None:
        return
