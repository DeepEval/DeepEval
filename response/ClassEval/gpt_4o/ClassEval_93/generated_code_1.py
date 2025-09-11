import numpy as np
from gensim import matutils
from numpy import dot, array
from numpy.linalg import norm

class VectorUtil:
    """
    The class provides vector operations, including calculating similarity, cosine similarities, average similarity, and IDF weights.
    """

    @staticmethod
    def similarity(vector_1, vector_2):
        """
        Compute the cosine similarity between one vector and another vector.
        """
        cos_sim = dot(vector_1, vector_2) / (norm(vector_1) * norm(vector_2))
        return cos_sim

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        """
        Compute cosine similarities between one vector and a set of other vectors.
        """
        cos_sims = [dot(vector_1, vec) / (norm(vector_1) * norm(vec)) for vec in vectors_all]
        return np.array(cos_sims)

    @staticmethod
    def n_similarity(vector_list_1, vector_list_2):
        """
        Compute cosine similarity between two sets of vectors.
        """
        mean_vec_1 = np.mean(vector_list_1, axis=0)
        mean_vec_2 = np.mean(vector_list_2, axis=0)
        cos_sim = dot(mean_vec_1, mean_vec_2) / (norm(mean_vec_1) * norm(mean_vec_2))
        return cos_sim

    @staticmethod
    def compute_idf_weight_dict(total_num, number_dict):
        """
        Calculate log(total_num+1/count+1) for each count in number_dict
        """
        return {key: np.log((total_num + 1) / (count + 1)) for key, count in number_dict.items()}


if __name__ == "__main__":
    # Test case for similarity
    vector_1 = np.array([1, 1])
    vector_2 = np.array([1, 0])
    output = VectorUtil.similarity(vector_1, vector_2)
    print(f"Similarity: {output}")  # Expected: 0.7071067811865475

    # Test case for cosine_similarities
    vector1 = np.array([1, 2, 3])
    vectors_all = [np.array([4, 5, 6]), np.array([7, 8, 9])]
    output = VectorUtil.cosine_similarities(vector1, vectors_all)
    print(f"Cosine Similarities: {output}")  # Expected: [0.97463185 0.95941195]

    # Test case for n_similarity
    vector_list1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    vector_list2 = [np.array([7, 8, 9]), np.array([10, 11, 12])]
    output = VectorUtil.n_similarity(vector_list1, vector_list2)
    print(f"n_similarity: {output}")  # Expected: 0.9897287473881233

    # Test case for compute_idf_weight_dict
    num_dict = {'key1': 0.1, 'key2': 0.5}
    output = VectorUtil.compute_idf_weight_dict(2, num_dict)
    print(f"IDF Weights: {output}")  # Expected: {'key1': 1.0033021088637848, 'key2': 0.6931471805599453}