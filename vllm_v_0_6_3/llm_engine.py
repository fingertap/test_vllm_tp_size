from __future__ import annotations

from vllm.engine.llm_engine import LLMEngine as vllm_LLMEngine
from vllm.engine.metrics_types import StatLoggerBase
from vllm.usage.usage_lib import UsageContext

from .arg_utils import EngineArgs


class LLMEngine(vllm_LLMEngine):
    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: dict[str, StatLoggerBase] | None = None,
    ):
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        executor_class = cls._get_executor_cls(engine_config)
        # Initialize the cluster and specify the executor class.
        assert (
            engine_config.device_config.device_type == "cuda"
        ), "Currently, the vllm in verl only support running on GPU"

        from .spmd_gpu_executor import SPMDGPUExecutor

        executor_class = SPMDGPUExecutor

        # Create the LLM engine.
        engine = cls(
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )
        return engine

    def init_cache_engine(self):
        self.model_executor.init_cache_engine()

    def free_cache_engine(self):
        self.model_executor.free_cache_engine()

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.model_executor.update_weight(name, dtype, shape, empty_cache)

    def meta(self):
        self.model_executor.meta()

    def cuda(self):
        self.model_executor.cuda()

    def cpu(self):
        self.model_executor.cpu()
