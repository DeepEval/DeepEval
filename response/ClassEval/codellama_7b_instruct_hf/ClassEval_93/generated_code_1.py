import numpy as np
from gensim import matutils
from numpy import dot, array

class VectorUtil:
    """
    The class provides vector operations, including calculating similarity, cosine similarities, average similarity, and IDF weights.
    """

    @staticmethod
    def similarity(vector_1, vector_2):
        """
        Compute the cosine similarity between one vector and another vector.
        :param vector_1: numpy.ndarray, Vector from which similarities are to be computed, expected shape (dim,).
        :param vector_2: numpy.ndarray, Vector from which similarities are to be computed, expected shape (dim,).
        :return: numpy.ndarray, Contains cosine distance between `vector_1` and `vector_2`
        >>> vector_1 = np.array([1, 1])
        >>> vector_2 = np.array([1, 0])
        >>> VectorUtil.similarity(vector_1, vector_2)
        0.7071067811865475
        """
        # Test case 1
        vector_1 = np.array([1, 1])
        vector_2 = np.array([1, 0])
        output = VectorUtil.similarity(vector_1, vector_2)
        expected_output = 0.7071067811865475
        assert output == expected_output

        # Test case 2
        vector_1 = np.array([1, 1, 1])
        vector_2 = np.array([1, 0, 0])
        output = VectorUtil.similarity(vector_1, vector_2)
        expected_output = 0.7071067811865475
        assert output == expected_output

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        """
        Compute cosine similarities between one vector and a set of other vectors.
        :param vector_1: numpy.ndarray, Vector from which similarities are to be computed, expected shape (dim,).
        :param vectors_all: list of numpy.ndarray, For each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).
        :return: numpy.ndarray, Contains cosine distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).
        >>> vector1 = np.array([1, 2, 3])
        >>> vectors_all = [np.array([4, 5, 6]), np.array([7, 8, 9])]
        >>> VectorUtil.cosine_similarities(vector1, vectors_all)
        [0.97463185 0.95941195]
        """
        # Test case 1
        vector1 = np.array([1, 2, 3])
        vectors_all = [np.array([4, 5, 6]), np.array([7, 8, 9])]
        output = VectorUtil.cosine_similarities(vector1, vectors_all)
        expected_output = [0.97463185, 0.95941195]
        assert np.allclose(output, expected_output)

        # Test case 2
        vector1 = np.array([1, 2, 3])
        vectors_all = [np.array([4, 5, 6]), np.array([7, 8, 9]), np.array([10, 11, 12])]
        output = VectorUtil.cosine_similarities(vector1, vectors_all)
        expected_output = [0.97463185, 0.95941195, 0.92731422]
        assert np.allclose(output, expected_output)

    @staticmethod
    def n_similarity(vector_list_1, vector_list_2):
        """
        Compute cosine similarity between two sets of vectors.
        :param vector_list_1: list of numpy vector
        :param vector_list_2: list of numpy vector
        :return: numpy.ndarray, Similarities between vector_list_1 and vector_list_2.
        >>> vector_list1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> vector_list2 = [np.array([7, 8, 9]), np.array([10, 11, 12])]
        >>> VectorUtil.n_similarity(vector_list1, vector_list2)
        0.9897287473881233
        """
        # Test case 1
        vector_list1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        vector_list2 = [np.array([7, 8, 9]), np.array([10, 11, 12])]
        output = VectorUtil.n_similarity(vector_list1, vector_list2)
        expected_output = 0.9897287473881233
        assert output == expected_output

        # Test case 2
        vector_list1 = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        vector_list2 = [np.array([10, 11, 12]), np.array([13, 14, 15]), np.array([16, 17, 18])]
        output = VectorUtil.n_similarity(vector_list1, vector_list2)
        expected_output = 0.9897287473881233
        assert output == expected_output

    @staticmethod
    def compute_idf_weight_dict(total_num, number_dict):
        """
        Calculate log(total_num+1/count+1) for each count in number_dict
        :param total_num: int
        :param number_dict: dict
        :return: dict
        >>> num_dict = {'key1':0.1, 'key2':0.5}
        >>> VectorUtil.compute_idf_weight_dict(2, num_dict)
        {'key1': 1.0033021088637848, 'key2': 0.6931471805599453}
        """
        # Test case 1
        num_dict = {'key1': 0.1, 'key2': 0.5}
        output = VectorUtil.compute_idf_weight_dict(2, num_dict)
        expected_output = {'key1': 1.0033021088637848, 'key2': 0.6931471805599453}
        assert output == expected_output

        # Test case 2
        num_dict = {'key1': 0.1, 'key2': 0.5, 'key3': 0.8}
        output = VectorUtil.compute_idf_weight_dict(2, num_dict)
        expected_output = {'key1': 1.0033021088637848, 'key2': 0.6931471805599453, 'key3': 0.4737512053778298}
        assert output == expected_output