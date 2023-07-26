#install dependencies
conda install -c conda-forge rdkit python=3.10 matplotlib scikit-learn numpy pandas seaborn jupyter
ipython kernel install --user --name=kdti

# Pytorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# torch_geometric
pip install torch_geometric

# Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
