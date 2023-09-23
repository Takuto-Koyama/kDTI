

from model import kDTI
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, DEBUG
from time import time 
from dataloader import DTIDataset 
from torch.utils.data import DataLoader 
from trainer import Trainer 
import torch 

import argparse 
import warnings, os 
import pandas as pd 
from config import Config
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="kDTI for DTI prediction")
parser.add_argument('-c', '--config', required=True, help="path to config file", type=str)
args = parser.parse_args()

def get_parser():
    parser = argparse.ArgumentParser(
        description='kDTI for DTI prediction',
        usage=f"python {os.path.basename(__file__)} -c config.yaml",
    )
    
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='debug mode'
    )

def get_logger(level, save_dir):
    logger = getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = False

    formatter = Formatter("%(asctime)s : %(levelname)s : %(message)s ")

    fh = FileHandler(filename=os.path.join(save_dir, "run.log"), mode='w')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    sh = StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def main():
    torch.cuda_empty_cache()
    warnings.filterwarnings("ignore", "invalid value encountered in divide")
    config = Config()
    with open(args.config, "r") as f:
        cfg= json.load(f)
    config.update_config(cfg)
    
    os.makedirs(config.output_path, exist_ok=True)
    
    log_level = DEBUG if args.debug else INFO
    logger = get_logger(log_level, config.output_path)
    
    logger.info(f"========== Configuration ==========")
    for k, v in config.items():
        logger.info(f"{k}: {v}")
    logger.info(f"GPU devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"===================================")