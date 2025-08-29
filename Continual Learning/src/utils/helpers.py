
import json
import numpy as np
import torch
from dataclasses import dataclass

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy arrays and other non-serializable types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16, np.float8)):
            return float(obj)
        if isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        if isinstance(obj, torch.Tensor):
            return bool(obj)
        if hasattr(obj, '__dict__'):
            #Try to serialize objects by their dict
            try:
                return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
            except:
                return str(obj)
        return super().default(obj)

@dataclass
class MergeConfig:
    """Configuration for intelligent merging"""
    similarity_threshold: float = 0.8
    trim_percentage: float = 0.1
    max_accuracy_drop: float = 2.0
    max_rejection_drop: float = 4.0
    max_fp_increase: float = 2.0
    ablation_samples: int = 1000
    promotion_threshold: float = 10.0