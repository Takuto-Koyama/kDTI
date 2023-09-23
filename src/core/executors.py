from abc import ABC, abstractmethod


from src.data.dataset import DTIDataset
from src.core.config import Config

class AbstractExecutor(ABC):
    def __init__(self, config: Config, logger):
        self.config = config 
        self.logger = logger
    
    def _create_dataset(self):
        raise NotImplementedError
    
    def run(self):
        raise NotImplementedError

class Trainer(AbstractExecutor):
    
    def __init__(self, config: Config, logger):
        super().__init__(config, logger)
        
        self.logger.info("Trainer initialized")
        self.dataset = None
        
        self.dataset = self._create_dataset()
        
    def _create_dataset(self):
        pass
    
    def run(self):
        pass

class Evaluator(AbstractExecutor):
    
    def __init__(self, config: Config, logger):
        super().__init__(config, logger)
    
    def run(self):
        pass

class Predictor(AbstractExecutor):
    
    def __init__(self, config: Config, logger):
        super().__init__(config, logger)
    
    def run(self):
        pass

class Pretrainer(AbstractExecutor):
    
    def __init__(self, config, Config, logger):
        super().__init__(config, logger)

    def run(self):
        pass