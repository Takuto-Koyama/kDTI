import torch.nn as nn

class kDTI(nn.Module):
    def __init__(self,  **config):
        super(KDTI, self).__init__()
        drug_module = config["DRUG"]["MODULE"]
        drug_readout = config["DRUG"]["READOUT"]
        protein_module = config["PROTEIN"]["MODULE"]
        pritein_flat = config["PROTEIN"]["FLAT"]
        out_features = config["DECODER"]["OUT_DIM"]
        
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]

        self.drug_module = create_class(drug_module)(**config["DRUG"])
        self.target_module = create_class(protein_module)(**config["PROTEIN"])
        
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
        
    def forward(self, input, mode="train"):
        v_d = self.drug_module(input)
        v_p = self.protein_module(input)
        f, att = self.bcb(v_d, v_p)
        score = self.mlp(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att
        

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x