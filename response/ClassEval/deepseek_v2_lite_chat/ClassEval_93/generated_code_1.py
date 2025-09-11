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
        """
        return dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        """
        Compute cosine similarities between one vector and a set of other vectors.
        :param vector_1: numpy.ndarray, Vector from which similarities are to be computed, expected shape (dim,).
        :param vectors_all: list of numpy.ndarray, For each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).
        :return: numpy.ndarray, Contains cosine distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).
        """
        return [self.similarity(vector_1, vector_v) for vector_v in vectors_all]

    @staticmethod
    def n_similarity(vector_list_1, vector_list_2):
        """
        Compute cosine similarity between two sets of vectors.
        :param vector_list_1: list of numpy vector
        :param vector_list_2: list of numpy vector
        :return: numpy.ndarray, Similarities between vector_list_1 and vector_list_2.
        """
        return [dot(vector_list_1[i], vector_list_2[i]) / (np.linalg.norm(vector_list_1[i]) * np.linalg.norm(vector_list_2[i])) for i in range(len(vector_list_1))]

    @staticmethod
    def compute_idf_weight_dict(total_num, number_dict):
        """
        Calculate log(total_num+1/count+1) for each count in number_dict
        :param total_num: int
        :param number_dict: dict
        :return: dict
        """
        idf_dict = {}
        for key, count in number_dict.items():
            if count > 0:
                idf_dict[key] = np.log(total_num + 1) - np.log(count + 1)
        return idf_dict

# Test cases
if __name__ == "__main__":
    # Test similarity
    vector_1 = np.array([1, 2, 3])
    vector_2 = np.array([4, 5, 6])
    print(VectorUtil.similarity(vector_1, vector_2))  # Should print a value close to 0.97

    # Test cosine_similarities
    vectors_all = [np.array([4, 5, 6]), np.array([7, 8, 9])]
    print(VectorUtil.cosine_similarities(vector_1, vectors_all))  # Should print [0.97463185, 0.95941195]

    # Test n_similarity
    vector_list_1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    vector_list_2 = [np.array([7, 8, 9]), np.array([10, 11, 12])]
    print(VectorUtil.n_similarity(vector_list_1, vector_list_2))  # Should print a list of values close to 1

    # Test compute_idf_weight_dict
    num_dict = {'key1': 0.1, 'key2': 0.5}
    total_num = 100
    print(VectorUtil.compute_idf_weight_dict(total_num, num_dict))  # Should print a dictionary with IDF values