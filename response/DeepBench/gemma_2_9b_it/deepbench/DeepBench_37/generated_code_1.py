import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import numpy as np
from graphconv.constants import GraphConvConstants

def generate_global_features(mol, features_generators):
    global_features = []
    dummy_mol = Chem.MolFromSmiles('CC')
    if mol.GetNumAtoms() == 0:
      global_features = [np.nan] * len(GraphConvConstants.FEATURE_GENERATORS)
    else:
      for generator in features_generators:
        if generator in GraphConvConstants.FEATURE_GENERATORS:
          if generator == 'MorganFingerprint':
            features = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()
          elif generator == 'AtomPairFingerprint':
            features = AllChem.GetAtomPairFingerprint(mol).ToBitString()
          elif generator == 'TopologicalTorsionFingerprint':
            features = AllChem.GetTopologicalTorsionFingerprint(mol).ToBitString()
          else:
            features = GraphConvConstants.FEATURE_GENERATORS[generator](mol)
          global_features.append(features)
      
      if len(global_features) == 0:
        global_features = [np.nan] * len(GraphConvConstants.FEATURE_GENERATORS)
      
      global_features_array = np.array(global_features)
      global_features_array = np.where(np.isnan(global_features_array), GraphConvConstants.REPLACE_TOKEN, global_features_array)
      return global_features_array 
    

if __name__ == "__main__":
    smiles = 'CCO'
    mol = Chem.MolFromSmiles(smiles)
    features_generators = ['MorganFingerprint', 'AtomPairFingerprint']
    global_features_array = generate_global_features(mol, features_generators)
    print(global_features_array)