import numpy as np
from sklearn.decomposition import DictionaryLearning

def dict_learning_online(X, n_components, alpha, max_iter, return_code=True):
    # Initialize dictionary learning
    model = DictionaryLearning(n_components=n_components, alpha=alpha, max_iter=max_iter, return_code=return_code)

    # Fit the model to the data
    model.fit(X)

    # Return the dictionary and code
    return model.components_, model.components_

if __name__ == "__main__":
    # Create sample input data
    X = np.random.rand(100, 50)

    # Set hyperparameters
    n_components = 10
    alpha = 1
    max_iter = 100

    # Call the dictionary learning function
    dictionary, code = dict_learning_online(X, n_components, alpha, max_iter)

    # Print the results
    print("Dictionary:")
    print(dictionary)
    print("Code:")
    print(code)