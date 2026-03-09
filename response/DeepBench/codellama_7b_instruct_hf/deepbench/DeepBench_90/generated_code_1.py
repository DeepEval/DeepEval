import numpy as np

def noise_matrix_is_valid(noise_matrix, py, verbose=False):
    """
    Checks if the given noise matrix is a learnable matrix.

    Parameters:
    noise_matrix (np.ndarray): Noise matrix
    py (np.ndarray): Prediction matrix
    verbose (bool, optional): Whether to print detailed information

    Returns:
    (bool) Whether the noise matrix is learnable
    """
    # Calculate the number of samples in the noise matrix
    num_samples = noise_matrix.shape[0]

    # Calculate the number of features in the noise matrix
    num_features = noise_matrix.shape[1]

    # Calculate the number of predictions in the prediction matrix
    num_predictions = py.shape[1]

    # Calculate the number of classes in the prediction matrix
    num_classes = np.unique(py).shape[0]

    # Calculate the entropy of the noise matrix
    noise_entropy = np.sum(-np.log(noise_matrix) * noise_matrix)

    # Calculate the entropy of the prediction matrix
    pred_entropy = np.sum(-np.log(py) * py)

    # Calculate the mutual information between the noise matrix and the prediction
  matrix
    mi = np.sum(noise_matrix * py)

    # Calculate the number of non-zero elements in the noise matrix
    non_zero_elements = np.count_nonzero(noise_matrix)

    # Calculate the number of non-zero elements in the prediction matrix
    pred_non_zero_elements = np.count_nonzero(py)

    # Calculate the number of non-zero elements in the intersection of the noise
  matrix and the prediction matrix
    intersection_non_zero_elements = np.count_nonzero(np.logical_and(noise_matrix,
  py))

    # Calculate the number of non-zero elements in the union of the noise matrix
  and the prediction matrix
    union_non_zero_elements = np.count_nonzero(np.logical_or(noise_matrix, py))

    # Calculate the number of non-zero elements in the difference of the noise
  matrix and the prediction matrix
    difference_non_zero_elements = np.count_nonzero(np.logical_xor(noise_matrix,
  py))

    # Calculate the number of non-zero elements in the symmetric difference of
  the noise matrix and the prediction matrix
    symmetric_difference_non_zero_elements = np.count_nonzero(np.logical_xor(noise_matrix,
  py))

    # Calculate the number of non-zero elements in the prediction matrix that are
  not in the noise matrix
    pred_non_zero_elements_not_in_noise = pred_non_zero_elements - intersection_non_zero_elements

    # Calculate the number of non-zero elements in the noise matrix that are not
  in the prediction matrix
    noise_non_zero_elements_not_in_pred = non_zero_elements - intersection_non_zero_elements

    # Calculate the number of non-zero elements in the prediction matrix that are
  not in the noise matrix
    pred_non_zero_elements_not_in_noise = pred_non_zero_elements - intersection_non_zero_elements

    # Calculate the number of non-zero elements in the noise matrix that are not
  in the prediction matrix
    noise_non_zero_elements_not_in_pred = non_zero_elements - intersection_non_zero_elements

    # Calculate the number of non-zero elements in the intersection of the prediction
  matrix and the noise matrix
    intersection_non_zero_elements = np.count_nonzero(np.logical_and(noise_matrix,
  py))

    # Calculate the number of non-zero elements in the union of the prediction
  matrix and the noise matrix
    union_non_zero_elements = np.count_nonzero(np.logical_or(noise_matrix, py))

    # Calculate the number of non-zero elements in the difference of the prediction
  matrix and the noise matrix
    difference_non_zero_elements = np.count_nonzero(np.logical_xor(noise_matrix,
  py))

    # Calculate the number of non-zero elements in the symmetric difference of
  the prediction matrix and the noise matrix
    symmetric_difference_non_zero_elements = np.count_nonzero(np.logical_xor(noise_matrix,
  py))

    # Calculate the number of non-zero elements in the prediction matrix that are
  not in the noise matrix
    pred_non_zero_elements_not_in_noise = pred_non_zero_elements - intersection_non_zero_elements

    # Calculate the number of non-zero elements in the noise matrix that are not
  in the prediction matrix
    noise_non_zero_elements_not_in_pred = non_zero_elements - intersection_non_zero_elements

    # Calculate the number of non-zero elements in the prediction matrix that are
  not in the noise matrix
    pred_non_zero_elements_not_in_noise = pred_non_zero_elements - intersection_non_zero_elements

    # Calculate the number of non-zero elements in the noise matrix that are not
  in the prediction matrix
    noise_non_zero_elements_not_in_pred = non_zero_elements - intersection_non_zero_elements

    # Calculate the number of non-zero elements in the intersection of the prediction
  matrix and the noise matrix
    intersection_non_zero_elements = np.count_nonzero(np.logical_and(noise_matrix,
  py))

    # Calculate the number of non-zero elements in the union of the prediction
  matrix and the noise matrix
    union_non_zero_elements = np.count_nonzero(np.logical_or(noise_matrix, py))

    # Calculate the number of non-zero elements in the difference of the prediction
  matrix and the noise matrix
    difference_non_zero_elements = np.count_nonzero(np.logical_xor(noise_matrix,
  py))

    # Calculate the number of non-zero elements in the symmetric difference of
  the prediction matrix and the noise matrix
    symmetric_difference_non_zero_elements = np.count_nonzero(np.logical_xor(noise_matrix,
  py))

    # Calculate the number of non-zero elements in the prediction matrix that are
  not in the noise matrix
    pred_non_zero_elements_not_in_noise = pred_non_zero_elements - intersection_non_zero_elements

    # Calculate the number of non-zero elements in the noise matrix that are not
  in the prediction matrix
    noise_non_zero_elements_not_in_pred = non_zero_elements - intersection_non_zero_elements

    # Calculate the number of non-zero elements in the prediction matrix that are
  not in the noise matrix
    pred_non_zero_elements_not_in_noise = pred_non_zero_elements - intersection_non_zero_elements

    # Calculate the number of non-zero elements in the noise matrix that are not
  in the prediction matrix
    noise_non_zero_elements_not_in_pred = non_zero_elements - intersection_non_zero_elements

    # Calculate the number of non-zero elements in the intersection of the prediction
  matrix and the noise matrix
    intersection_non_zero_elements = np.count_nonzero(np.logical_and(noise_matrix,
  py))

    # Calculate the number of non-zero elements in the union of the prediction
  matrix and the noise matrix
    union_non_zero_elements = np.count_nonzero(np.logical_or(noise_matrix, py))

    # Calculate the number of non-zero elements in the difference of the prediction
  matrix and the noise matrix
    difference_non_zero_elements = np.count_nonzero(np.logical_xor(noise_matrix,
  py))

    # Calculate the number of non-zero elements in the symmetric difference of
  the prediction matrix and the noise matrix
    symmetric_difference_non_zero_elements = np.count_nonzero(np.logical_xor(noise_matrix,
  py))

    # Calculate the number of non-zero elements in the prediction matrix that are
  not