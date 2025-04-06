from abc import ABC, abstractmethod
import os
from .config import RESULTS_DIR
from typing import Dict, Any, Optional


class BenchmarkRunner(ABC):
    def __init__(self, model_id: str, hf_token: Optional[str] = None):
        self.model_id = model_id
        self.hf_token = hf_token
        self.results_dir = os.path.join(RESULTS_DIR, self.framework_name())
        os.makedirs(self.results_dir, exist_ok=True)

    @staticmethod
    @abstractmethod
    def framework_name() -> str:
        pass

    @abstractmethod
    def run(self, task_details: Dict[str, Any], backend: str, **kwargs) -> bool:
        pass
