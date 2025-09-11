import numpy as np
from numpy import dot, array, linalg

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
        :return: float, Contains cosine distance between `vector_1` and `vector_2`
        """
        if np.all(vector_1 == 0) or np.all(vector_2 == 0):
            return 0.0
        return dot(vector_1, vector_2) / (linalg.norm(vector_1) * linalg.norm(vector_2))

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        """
        Compute cosine similarities between one vector and a set of other vectors.
        :param vector_1: numpy.ndarray, Vector from which similarities are to be computed, expected shape (dim,).
        :param vectors_all: list of numpy.ndarray, For each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).
        :return: numpy.ndarray, Contains cosine distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).
        """
        similarities = np.array([VectorUtil.similarity(vector_1, vector) for vector in vectors_all])
        return similarities

    @staticmethod
    def n_similarity(vector_list_1, vector_list_2):
        """
        Compute cosine similarity between two sets of vectors.
        :param vector_list_1: list of numpy vector
        :param vector_list_2: list of numpy vector
        :return: numpy.ndarray, Similarities between vector_list_1 and vector_list_2.
        """
        total_similarity = 0
        count = 0
        for vec1 in vector_list_1:
            for vec2 in vector_list_2:
                total_similarity += VectorUtil.similarity(vec1, vec2)
                count += 1
        return total_similarity / count if count > 0 else 0.0

    @staticmethod
    def compute_idf_weight_dict(total_num, number_dict):
        """
        Calculate log(total_num+1/count+1) for each count in number_dict
        :param total_num: int
        :param number_dict: dict
        :return: dict
        """
        idf_weights = {}
        for key, count in number_dict.items():
            idf_weights[key] = np.log((total_num + 1) / (count + 1))
        return idf_weights


# Test cases
if __name__ == "__main__":
    # Test case for similarity
    vector_1 = np.array([1, 1])
    vector_2 = np.array([1, 0])
    output_similarity = VectorUtil.similarity(vector_1, vector_2)
    print("Cosine Similarity:", output_similarity)

    # Test case for cosine_similarities
    vector1 = np.array([1, 2, 3])
    vectors_all = [np.array([4, 5, 6]), np.array([7, 8, 9])]
    output_cosine_similarities = VectorUtil.cosine_similarities(vector1, vectors_all)
    print("Cosine Similarities:", output_cosine_similarities)

    # Test case for n_similarity
    vector_list1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    vector_list2 = [np.array([7, 8, 9]), np.array([10, 11, 12])]
    output_n_similarity = VectorUtil.n_similarity(vector_list1, vector_list2)
    print("N Similarity:", output_n_similarity)

    # Test case for compute_idf_weight_dict
    num_dict = {'key1': 0.1, 'key2': 0.5}
    output_idf_weights = VectorUtil.compute_idf_weight_dict(2, num_dict)
    print("IDF Weights:", output_idf_weights)