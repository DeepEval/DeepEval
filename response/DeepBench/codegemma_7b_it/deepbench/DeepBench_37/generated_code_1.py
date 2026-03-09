import numpy as np
from rdkit import Chem
from rdkit.Chem import GraphConvConstants

def generate_global_features(mol, features_generators):
    global_features = []
    for generator in features_generators:
        if generator in GraphConvConstants.FEATURE_GENERATORS:
            featurizer = generator()
            features = featurizer.featurize(mol)
            global_features.extend(features)
    if not mol.GetNumHeavyAtoms():
        mol = Chem.MolFromSmiles("CC")
    global_features_array = np.array(global_features, dtype=np.float32)
    global_features_array[np.isnan(global_features_array)] = -999.0
    return global_features_array

if __name__ == "__main__":
    smiles = "C(=O)Nc1ccccc1C(=O)Nc1ccccc1"
    mol = Chem.MolFromSmiles(smiles)

    # Example feature generators
    features_generators = [
        GraphConvConstants.AtomFeaturesGenerator(),
        GraphConvConstants.BondFeaturesGenerator(),
    ]

    global_features_array = generate_global_features(mol, features_generators)

    print(global_features_array)