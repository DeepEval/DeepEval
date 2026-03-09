import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from graphconv_constants import FEATURE_GENERATORS

def generate_global_features(mol, features_generators):
    global_features = []
    
    if mol.GetNumHeavyAtoms() == 0:
        mol = Chem.MolFromSmiles('C')  # Use methane as a dummy molecule
    
    for generator in features_generators:
        if generator in FEATURE_GENERATORS:
            features = FEATURE_GENERATORS[generator](mol)
            global_features.extend(features)
    
    global_features_array = np.array(global_features)
    global_features_array = np.where(np.isnan(global_features_array), replace_token, global_features_array)
    
    return global_features_array

# Example usage
if __name__ == "__main__":
    mol = Chem.MolFromSmiles('CCO')  # Ethanol molecule
    features_generators = ['Morgan', 'RDKit']
    replace_token = 0
    result = generate_global_features(mol, features_generators, replace_token)
    print(result)