import numpy as np
from rouge import rouge_l_sentence_level


class SetBasedSimilarity:
    """
    Set-based similarity measures are based on the overlap between two sets of items.
    """
    def rouge_set(self, candidate, reference):

        if type(reference) == list and type(candidate) == list:
            set_candidate = set(candidate)
            set_reference = set(reference)
            intersection = set_candidate.intersection(set_reference)

            recall = len(intersection) / len(set_reference)
            precision = len(intersection) / len(set_candidate)
            if recall + precision == 0:
                f1_score = 0.0
            else: 
                f1_score = 2 * recall * precision / (recall + precision) 

        return recall, precision, f1_score
    
    def main(self, candidate, reference):
        """
        Calculate the similarity of two API call sets based on the intersection between the two sets.
        """
        recall, precision, f1_score = self.rouge_set(candidate=candidate, reference=reference)

        similarity_dict = {
            "ReCall": recall,
            "Precision": precision,
            "F1_Score": f1_score
        }

        return similarity_dict


class LCSBasedSimilarity:
    """
    Calculate the similarity of two segments of text based on the longest common subsequence. rouge_l: Calculate ROUGE-L similarity based on LCS
    Note: Because the definitions of recall and precsion are different, the input of each similarity calculation function needs to be configured in the order of reference and candidate
    """
    def rouge_l(self, candidate, reference):
        if isinstance(reference, list) and isinstance(candidate, list):
            recall, precision, f1_score = rouge_l_sentence_level(summary_sentence=candidate,
                                                                reference_sentence=reference)
            similarity = {
                "ReCall": recall,
                "Precision": precision,
                "F1_Score": f1_score,
            }
        else:
            print("error in rouge-l")
            similarity = {'ReCall': 0, 'Precision': 0, 'F1_Score': 0}

        return similarity



class APISetSimilarity(SetBasedSimilarity):
    """
    Calculate the similarity of two API call sets
    """

    def __init__(self, metric):
        self.metric = metric

    def compute_api_set_similarity(self, candidate, reference):

        try:
            if self.metric == 'set_based':
                similarity = self.main(candidate=candidate, reference=reference)
            else:
                print("error in compute_api_set_similarity")
                similarity = {'Recall': 0, 'Precision': 0, 'F1_Score': 0}
                
        except Exception as e:
            print("error in compute_api_set_similarity:", e)
            raise
        return similarity


class APISequenceSetsSimilarity(LCSBasedSimilarity):
    """
    Calculate the similarity of two sets of API sequences, each set including multiple API sequences
    """

    def __init__(self, metric):
        self.metric = metric

    def matrix_api_seq_similarity(self, reference, candidate):
        """
        Calculate the similarity of two sets of API sequences by computing the similarity of each API sequence, then arranging the similarity into a two-dimensional matrix, and calculating the average similarity of the two API sequence
        """
        try:
            m = len(reference)
            n = len(candidate)
            if m != 0 and n != 0:
                recall_matrix, precision_matrix, f1score_matrix = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))
                for i in range(m):
                    for j in range(n):
                        if self.metric == 'rouge_l':
                            similarity = self.rouge_l(candidate=candidate[j], reference=reference[i])
                        else:
                            print("error in compute_api_seq_set_similarity")
                            similarity = 0

                        recall_matrix[i][j] = similarity['ReCall']
                        precision_matrix[i][j] = similarity['Precision']
                        f1score_matrix[i][j] = similarity['F1_Score']

                api_seq_set_recall = (np.mean(np.max(recall_matrix, axis=1)) + np.mean(np.max(recall_matrix, axis=0))) / 2
                api_seq_set_precision = (np.mean(np.max(precision_matrix, axis=1)) + np.mean(np.max(precision_matrix, axis=0))) / 2
                api_seq_set_f1score = (np.mean(np.max(f1score_matrix, axis=1)) + np.mean(np.max(f1score_matrix, axis=0))) / 2

                similarity_dict = {
                    "ReCall": api_seq_set_recall,
                    "Precision": api_seq_set_precision,
                    "F1_Score": api_seq_set_f1score
                }

            else:
                similarity_dict = {"Recall": 0,"Precision": 0, "F1_Score": 0}

        except Exception as e:
            print("error in matrix_api_seq_similarity:", e)
            raise

        return similarity_dict