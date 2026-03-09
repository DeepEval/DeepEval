import numpy as np
from rdkit import Chem
from rdkit.Chem.FeatMaps import GraphConvConstants

def generate_global_features(mol, features_generators):
    # Check if the molecule has any heavy atoms
    if mol.GetNumHeavyAtoms() == 0:
        # Use a dummy molecule (methane) to determine the length of the features
        dummy_mol = Chem.MolFromSmiles('CC')
        num_features = GraphConvConstants.FEATURE_GENERATORS[features_generators[0]](dummy_mol)
    else:
        # Get the number of features from the first generator
        num_features = GraphConvConstants.FEATURE_GENERATORS[features_generators[0]](mol)

    # Initialize the global features array
    global_features = np.zeros((num_features,))

    # Loop through the feature generators and featurize the molecule
    for generator in features_generators:
        # Check if the generator is available in the GraphConvConstants
        if generator in GraphConvConstants.FEATURE_GENERATORS:
            # Featurize the molecule using the generator
            features = GraphConvConstants.FEATURE_GENERATORS[generator](mol)
            # Append the features to the global_features list
            global_features = np.append(global_features, features)

    # Convert the global_features list to a numpy array
    global_features_array = np.array(global_features)

    # Replace any NaN values with a specified replace token
    global_features_array[np.isnan(global_features_array)] = 0

    return global_features_array

if __name__ == "__main__":
    # Create sample input values
    mol = Chem.MolFromSmiles('CC(=O)Nc1ccc(cc1)S(=O)(=O)N')
    features_generators = ['AtomFeatures', 'BondFeatures']

    # Call the function
    global_features_array = generate_global_features(mol, features_generators)

    # Print the results
    print(global_features_array)