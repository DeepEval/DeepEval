import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

class GraphConvConstants:
    FEATURE_GENERATORS = {
        "MorganFingerprint": lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024),
        "TopologicalTorsion": lambda mol: AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=1024),
        "RDKFingerprint": lambda mol: Chem.RDKFingerprint(mol)
    }

def generate_global_features(mol, features_generators, replace_token=0):
    global_features = []

    # Check if molecule has heavy atoms
    if mol.GetNumHeavyAtoms() == 0:
        mol = Chem.MolFromSmiles("C")  # Methane as dummy molecule

    for generator_name in features_generators:
        if generator_name in GraphConvConstants.FEATURE_GENERATORS:
            generator = GraphConvConstants.FEATURE_GENERATORS[generator_name]
            features = generator(mol)
            global_features.extend(features)

    global_features_array = np.array(global_features, dtype=float)
    global_features_array = np.nan_to_num(global_features_array, nan=replace_token)

    return global_features_array

if __name__ == "__main__":
    smiles = "CCO"  # Example molecule: Ethanol
    mol = Chem.MolFromSmiles(smiles)
    features_generators = ["MorganFingerprint", "RDKFingerprint"]
    
    result = generate_global_features(mol, features_generators)
    print(result)