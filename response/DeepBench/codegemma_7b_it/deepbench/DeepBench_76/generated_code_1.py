import numpy as np

def moa(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    """
    Maximization of Average ensemble method for combining multiple estimators.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        Score matrix outputted from various estimators.
    n_buckets : int, default 5
        Number of subgroups to build.
    method : str, default 'static'
        Method to build subgroups ('static' or 'dynamic').
    bootstrap_estimators : bool, default False
        Whether estimators are drawn with replacement.
    random_state : int, RandomState instance, or None, default None
        Seed for the random number generator.

    Returns
    -------
    combined_scores : numpy array of shape (n_samples,)
        Combined outlier scores.
    """

    # Call the combo_moa function with the specified parameters
    combined_scores = combo_moa(scores, n_buckets, method, bootstrap_estimators, random_state)

    return combined_scores

if __name__ == "__main__":
    # Sample input values
    scores = np.array([[0.5, 0.2, 0.8], [0.3, 0.7, 0.9], [0.6, 0.4, 0.7]])
    n_buckets = 3

    # Call the moa function
    combined_scores = moa(scores, n_buckets)

    # Print the results
    print(combined_scores)