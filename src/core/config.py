
from typing import Literal, Optional, List, Dict, Any, DefaultDict
import torch 


class Config:
    
    enabled_gpus: List[int]
    output_path: str
    
    
    def get_device(self) -> torch.device:
        device_id = self.enabled_gpus[0] if len(self.enabled_gpus) == 1 else 0
        device_name = "cuda:" + str(device_id) if torch.cuda.is_available() and self.use_cuda else "cpu"
        
        return torch.device(device_name)
    
    def update_config(self, cfg):
        for k, v in cfg.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise KeyError(f"Invalid key {k} in config file")
    
    