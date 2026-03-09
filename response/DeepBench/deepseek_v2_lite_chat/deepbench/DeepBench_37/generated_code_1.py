from rdkit import Chem
import numpy as np

# Dictionary mapping feature names to their corresponding RDKit generators
FEATURE_GENERATORS = {
    "heavy_atom_count": Chem.rdPartialCharges.HeavyAtomSubstructure,
    "atom_count": Chem.rdchem.GetTotalAtomCount
}

def generate_global_features(mol, features_generators):
    global_features = []
    
    # Check if the molecule has heavy atoms
    if not mol.GetNumAtoms():
        # Use a dummy molecule (methane) to determine the length of the features
        dummy_mol = Chem.MolFromSmiles('C')
        if dummy_mol is None:
            raise ValueError("Failed to parse methane SMILES")
        heavy_atom_count = dummy_mol.GetNumAtoms()
    else:
        heavy_atom_count = mol.GetNumAtoms()
    
    # Loop through the list of feature generators
    for generator in features_generators:
        if generator in FEATURE_GENERATORS:
            feature_func = FEATURE_GENERATORS[generator]
            mol.GetNumConformers()  # Ensure conformers are loaded
            feature = feature_func(mol)
            global_features.append(feature)
    
    # Determine the length of the global features if not provided
    if not global_features:
        raise ValueError("No feature generators specified or applied")
    feature_length = max(len(feat) for feat in global_features)
    
    # Convert list to numpy array and replace NaN values with a specified token
    global_features_array = np.array(global_features, dtype=object)
    if np.isnan(replace_token).any():
        # Ensure replace_token is not a NaN value
        replace_token = replace_token if not np.isnan(replace_token) else 0
        global_features_array = np.nan_to_num(global_features_array, nan=replace_token)
    
    return global_features_array

# Minimal runnable example
if __name__ == "__main__":
    from rdkit import DataStructs
    from rdkit import Chem

    # Sample RDKit molecule
    mol = Chem.MolFromSmiles('C')
    if mol is None:
        raise ValueError("Failed to parse SMILES")

    # Sample list of feature generators
    features_generators = ["heavy_atom_count", "atom_count"]

    # Generate global features
    global_features_array = generate_global_features(mol, features_generators)

    # Print results
    print("Global Features Array:", global_features_array)