import numpy as np
from rdkit import Chem
import GraphConvConstants

def generate_global_features(mol, features_generators, replace_token=np.nan):
    global_features = []
    dummy_molecule = Chem.MolFromSmiles('CH4')  # Methane

    for feature_generator in features_generators:
        if feature_generator in GraphConvConstants.FEATURE_GENERATORS:
            features = GraphConvConstants.FEATURE_GENERATORS[feature_generator](mol)
            global_features.append(features)

    if len(global_features) == 0:
        # If the molecule has no heavy atoms, use the dummy molecule to determine the length of the features
        for feature_generator in features_generators:
            if feature_generator in GraphConvConstants.FEATURE_GENERATORS:
                features = GraphConvConstants.FEATURE_GENERATORS[feature_generator](dummy_molecule)
                global_features.append(features)

    global_features_array = np.array(global_features)
    
    # Replace any NaN values with the specified replace_token value
    global_features_array[np.isnan(global_features_array)] = replace_token
    
    return global_features_array

if __name__ == "__main__":
    # Create a sample molecule
    mol = Chem.MolFromSmiles('CCO')

    # Define a sample feature generator
    def sample_feature_generator(mol):
        return [1, 2, 3]

    # Create a list of feature generators
    features_generators = ['sample_feature_generator']

    # Call the function and print the results
    global_features_array = generate_global_features(mol, features_generators)
    print(global_features_array)