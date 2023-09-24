import pandas as pd

import torch
from torch_geometric.data import Dataset, Data
from src.data.featurizers import *

class DTIDataset(Dataset):
    def __init__(self, config, list_ids, df):
        self.config = config
        self.list_ids = list_ids 
        self.df = df
        self.target_columns = config.target_columns
        
        self.drug_featurizer = GraphFeaturizer()
        self.target_featurizer = Tokenizer()
        
    def __len__(self):
        return len(self.list_ids)
    
    def __getitem__(self, index):
        index = self.list_ids[index]
        smiles = self.df.iloc[index]["smiles"]
        drug_features = self.drug_featurizer(smiles)
        target_sequence = self.df.iloc[index]["target_sequence"]
        target_features = self.target_featurizer(target_sequence)
        y = self.df.iloc[index][self.target_columns]
        y = torch.Tensor([y])
        
        data = Data(drug_features=drug_features, 
                    smiles=smiles, 
                    target_features=target_features,
                    target_sequence=target_sequence,
                    y=y
                    )
        
        return data 