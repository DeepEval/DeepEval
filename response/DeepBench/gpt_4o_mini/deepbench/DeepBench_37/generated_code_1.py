import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from some_module import GraphConvConstants  # Replace with the actual module containing GraphConvConstants

def generate_global_features(mol, features_generators, replace_token=0.0):
    global_features = []
    
    if mol.GetNumHeavyAtoms() == 0:
        dummy_mol = Chem.MolFromSmiles('C')  # Methane as the dummy molecule
        mol = dummy_mol
    
    for feature_generator in features_generators:
        if feature_generator in GraphConvConstants.FEATURE_GENERATORS:
            features = feature_generator(mol)
            global_features.append(features)
    
    global_features_array = np.array(global_features)
    global_features_array = np.nan_to_num(global_features_array, nan=replace_token)
    
    return global_features_array

if __name__ == "__main__":
    # Sample input for validation
    sample_smiles = 'CCO'  # Ethanol
    mol = Chem.MolFromSmiles(sample_smiles)
    features_generators = [some_feature_generator_1, some_feature_generator_2]  # Replace with actual generators

    global_features = generate_global_features(mol, features_generators)
    print(global_features)