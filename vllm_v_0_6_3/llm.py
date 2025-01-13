from __future__ import annotations

from vllm import LLM as vllm_LLM
from vllm.utils import Counter

from .arg_utils import EngineArgs
from .llm_engine import LLMEngine


class LLM(vllm_LLM):
    def __init__(self, model: str, **kwargs):
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        kwargs["enforce_eager"] = True
        engine_args = EngineArgs(model=model, **kwargs)
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

    def free_cache_engine(self):
        self.llm_engine.free_cache_engine()

    def init_cache_engine(self):
        self.llm_engine.init_cache_engine()

    def cpu(self):
        self.llm_engine.cpu()

    def meta(self):
        self.llm_engine.meta()

    def cuda(self):
        self.llm_engine.cuda()

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.llm_engine.update_weight(name, dtype, shape, empty_cache)
