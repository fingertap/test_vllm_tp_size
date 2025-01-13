from dataclasses import dataclass

from vllm.engine.arg_utils import EngineArgs as vllm_EngineArgs


@dataclass
class EngineArgs(vllm_EngineArgs):
    def __post_init__(self):
        # We require the tokenizer be provided explicitly
        if self.tokenizer is None:
            if isinstance(self.model, str):
                self.tokenizer = self.model
            else:
                raise ValueError("The tokenizer must be provided explicitly when `model` is not a valid path!")
