


class SimpleFeaturizer:
    def __init__(self, inputs, outputs, should_cache=False, rewrite=True):
        self.inputs = inputs
        self.outputs = outputs
        self.should_cache = should_cache
        self.rewrite = rewrite
        self.cache = {}

    def _process(self, data, entry):
        return data  # This should be overridden in subclasses

    def run(self, data):
        try:
            if len(self.inputs) != len(self.outputs):
                print("Inputs and outputs must have the same length.")
                return

            for index in range(len(self.inputs)):
                raw_data = data.inputs[self.inputs[index]]
                if self.rewrite:
                    del data.inputs[self.inputs[index]]

                result = self.process(raw_data, data) if not self.should_cache or raw_data not in self.cache else self.cache[raw_data]
                if self.should_cache:
                    self.cache[raw_data] = result

                data.inputs[self.outputs[index]] = result
        except (ValueError, IndexError, AttributeError, TypeError) as e:
            print(f"Warning: Could not run featurizer on {data.id_} --- {e}")



class SimpleTorchGeometricFeaturizer(SimpleFeaturizer):
    
    def _process(self, data):
        mol = Chem.MolFromSmiles(data)
        if mol is None:
            print("Could not featurize entry: [{}]".format(data))
            return None

        atom_features = self._get_vertex_features(mol)
        atom_features = torch.FloatTensor(atom_features).view(-1, len(atom_features[0]))

        edge_indices, edge_attributes = self._get_edge_features(mol)
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        edge_attributes = torch.FloatTensor(edge_attributes)

        if edge_indices.numel() > 0:  # Sort indices
            permutation = (edge_indices[0] * atom_features.size(0) + edge_indices[1]).argsort()
            edge_indices, edge_attributes = edge_indices[:, permutation], edge_attributes[permutation]

        return TorchGeometricData(x=atom_features, edge_index=edge_indices, edge_attr=edge_attributes, smiles=data)

    def _get_vertex_features(self, mol):
        return [self._featurize_atom(atom) for atom in mol.GetAtoms()]

    def _get_edge_features(self, mol):
        edge_indices, edge_attributes = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_indices += [[i, j], [j, i]]
            bond_features = self._featurize_bond(bond)
            edge_attributes += [bond_features, bond_features]

        return edge_indices, edge_attributes

    def _featurize_atom(self, atom):
        return []  # This should be overridden in subclasses

    def _featurize_bond(self, bond):
        return []  # This should be overridden in subclasses

