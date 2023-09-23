
from logging import INFO, DEBUG
from time import time 
import torch 

import argparse 
import warnings, os 

from src.core.executors import *
from src.core.config import Config
from src.utils.loggers import get_logger

def get_parser():
    parser = argparse.ArgumentParser(
        description='kDTI for DTI prediction',
        usage=f"python {os.path.basename(__file__)} -c config.yaml",
    )
    
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='path to config file'
    )

def main():
    start_time = time()
    parser = argparse.ArgumentParser(description="kDTI for DTI prediction")
    parser.add_argument('-c', '--config', required=True, help="path to config file", type=str)
    parser.add_argument('-d', '--debug', action='store_true', help="debug mode")
    args = parser.parse_args()
    
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", "invalid value encountered in divide")
    
    config = Config(args.config)
    
    os.makedirs(config.output_path, exist_ok=True)
    
    mode_dict = {"train": Trainer,
                "evaluate": Evaluator,
                "predict": Predictor,
                "pretrain": Pretrainer}
    
    
    log_level = DEBUG if args.debug else INFO
    logger = get_logger(log_level, config.output_path)
    
    executor = mode_dict[config.mode](config, logger)
    executor.run()
    
    end_time = time()
    logger.info("Total time: {:.2f} min".format((end_time - start_time) / 60))
    
if __name__ == "__main__":
    main()