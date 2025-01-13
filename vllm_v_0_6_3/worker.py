from __future__ import annotations

import os
from typing import Type

import torch
import vllm.envs as envs
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    PromptAdapterConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm.distributed import init_distributed_environment, set_custom_all_reduce
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, IntermediateTensors, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner
from vllm.worker.model_runner_base import ModelRunnerInputBase
from vllm.worker.worker import Worker as vllm_Worker
from vllm.worker.worker import _check_if_gpu_supports_dtype
from vllm.worker.worker_base import WorkerInput

from .parallel_state import ensure_model_parallel_initialized


class Worker(vllm_Worker):
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: LoRAConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prompt_adapter_config: PromptAdapterConfig | None = None,
        is_driver_worker: bool = False,
        model_runner_cls: Type[GPUModelRunnerBase] | None = None,
        observability_config: ObservabilityConfig | None = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.parallel_config.rank = rank
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.prompt_adapter_config = prompt_adapter_config
        self.is_driver_worker = True
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()
        self.observability_config = observability_config

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_args = (
            {}
            if speculative_config is None
            or (speculative_config.draft_model_config.model == model_config.model)
            or (speculative_config.draft_model_config.hf_config.model_type not in ["medusa", "mlp_speculator"])
            else {"return_hidden_states": True}
        )

        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_runner_cls is not None:
            ModelRunnerClass = model_runner_cls
        elif self.model_config.embedding_mode:
            ModelRunnerClass = EmbeddingModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=self.is_driver_worker,
            prompt_adapter_config=prompt_adapter_config,
            observability_config=observability_config,
            **speculative_args,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: list[CacheEngine] = None
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.gpu_cache: list[list[torch.Tensor]] | None = None
        self._seq_group_metadata_cache: dict[str, SequenceGroupMetadata] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            from loguru import logger

            logger.info("Profiling enabled. Traces will be saved to: %s", torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(torch_profiler_trace_dir, use_gzip=True),
            )
        else:
            self.profiler = None

    def init_device(self):
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

            self.rank = self.rank if self.rank is not None else int(os.getenv("RANK", "-1"))
            # NOTE: each worker only sees one GPU provided by ray
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)

            # Use the world_size set by ray
            world_size = int(os.getenv("WORLD_SIZE", "-1"))
            assert world_size != -1, "WORLD_SIZE is not set!"
            self.parallel_config.world_size = world_size

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.parallel_config, self.rank, self.distributed_init_method, self.local_rank
        )
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def execute_model(
        self, execute_model_req: ExecuteModelRequest, intermediate_tensors: IntermediateTensors | None = None
    ) -> list[SamplerOutput] | None:
        """
        Execute model in Single Program Multiple Data (SPMD) fashion.
        All workers take the same request, prepare the input and
        execute the model.
        """
        assert execute_model_req is not None, (
            "_execute_model_spmd() requires each worker to take in an " "ExecuteModelRequest"
        )
        worker_input: WorkerInput = self.prepare_worker_input(execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = self.model_runner.prepare_model_input(
            execute_model_req.seq_group_metadata_list
        )

        # verl.worker.workerbase.WorkerBase
        # swap cache
        super().execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        return self.model_runner.execute_model(
            model_input,
            self.kv_cache[worker_input.virtual_engine] if self.kv_cache is not None else None,
            intermediate_tensors,
        )

    def free_cache_engine(self):
        assert self.model_config.enforce_eager, "Must use eager mode to offload!"
        self.cache_engine = None
        self.gpu_cache = None
        torch.cuda.empty_cache()

    def cpu(self):
        assert self.model_config.enforce_eager, "Must use eager mode to offload!"
        for param in self.model_runner.model.parameters():
            param.meta_tensor = param.data.to("cpu", non_blocking=True)

        self.free_cache_engine()

    def meta(self):
        assert self.model_config.enforce_eager, "Must use eager mode to offload!"
        for param in self.model_runner.model.parameters():
            param.meta_tensor = param.data.to("meta")
            param.data = torch.Tensor([])

        self.free_cache_engine()

    def cuda(self):
        assert self.model_config.enforce_eager, "Must use eager mode to offload!"
        for param in self.model_runner.model.parameters():
            if not len(param.data):
                # Reload from meta device, we just to_empty
                param.data = torch.empty_like(param.meta_tensor, device="cuda")
                param.meta_tensor = None
            else:
                param.data = param.data.to("cuda", non_blocking=True)

        # NOTE: we do not init cache engine here, as this moment some other
        #       worker may be offloading, and we do not want OOM issues.

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        # if torch.distributed.get_rank() == 0:
        #     print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: str | None = "env://",
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank, distributed_init_method, local_rank)

    ensure_model_parallel_initialized(
        tensor_model_parallel_size=parallel_config.tensor_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_parallel_size,
    )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
