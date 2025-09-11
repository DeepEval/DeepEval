import numpy as np
from rouge import rouge_l_sentence_level
from nltk.metrics import edit_distance # Levenshtein Distance
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction


class SetBasedSimilarity:
    def dice_similarity(self, candidate, reference):
        if type(candidate) == list and type(reference) == list:
            set_candidate = set(candidate)
            set_reference = set(reference)
            intersection = set_candidate.intersection(set_reference)
            if len(set_candidate) + len(set_reference) == 0:
                return 0.0 
            dice_similarity = 2 * len(intersection) / (len(set_candidate) + len(set_reference))
            similarity = {
                "similarity": dice_similarity
            }
        else:
            print("error in dice_similarity")
            similarity = {"similarity": 0}
        
        return similarity

    def bleu(self, candidate, reference):
        reference = [reference] 
        if isinstance(reference[0], list) and isinstance(candidate, list):
            weights = (1, 0, 0, 0) 
            smooth_fn = SmoothingFunction().method1
            score = sentence_bleu(reference, candidate, weights=weights, smoothing_function=smooth_fn)
            similarity = {
                "similarity": score
            }
        else:
            print("error in bleu_similarity")
            similarity = {'similarity': 0}
        
        return similarity


class SeqBasedSimilarity:
    """
    Calculate the similarity of two segments of text based on the longest common subsequence. rouge_l: Calculate ROUGE-L similarity based on LCS
    Note: Because the definitions of recall and precsion are different, the input of each similarity calculation function needs to be configured in the order of reference and candidate
    """
    def bleu(self, candidate, reference):
        reference = [reference] 
        if isinstance(reference[0], list) and isinstance(candidate, list):
            weights = (1, 0, 0, 0) 
            smooth_fn = SmoothingFunction().method1
            score = sentence_bleu(reference, candidate, weights=weights, smoothing_function=smooth_fn)
            similarity = {
                "similarity": score
            }
        else:
            print("error in bleu_similarity")
            similarity = {'similarity': 0}
        
        return similarity
    
    def edit_distance(self, candidate, reference):
        """
        Levenshtein Distance Similarity
        """
        if isinstance(reference, list) and isinstance(candidate, list):
            distance = edit_distance(s1=candidate, s2=reference) 
            similarity = 1 - (distance / max(len(candidate), len(reference)))
            similarity = {
                "similarity": similarity
            }
        else:
            print("error in edit_distance_similarity")
            similarity = {"similarity": 0}

        return similarity

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
            if self.metric == 'dice':
                similarity = self.dice_similarity(candidate=candidate, reference=reference)
            elif self.metric == 'bleu':
                similarity = self.bleu(candidate=candidate, reference=reference)
            else:
                print("error in compute_api_set_similarity")
                similarity = {"similarity": 0}
                
        except Exception as e:
            print("error in compute_api_set_similarity:", e)
            raise
        return similarity


class APISequenceSetsSimilarity(SeqBasedSimilarity):
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
                        if self.metric == 'edit_distance':
                            similarity = self.edit_distance(candidate=candidate[j], reference=reference[i])
                        elif self.metric == 'bleu':
                            similarity = self.bleu(candidate=candidate[j], reference=reference[i])
                        elif self.metric == 'rouge_l':
                            similarity = self.rouge_l(candidate=candidate[j], reference=reference[i])
                        else:
                            print("error in compute_api_seq_set_similarity")
                            similarity = 0

                        similarity_matrix[i][j] = similarity['similarity']

                api_seq_set_similarity = (np.mean(np.max(similarity_matrix, axis=1)) + np.mean(np.max(similarity_matrix, axis=0))) / 2
                similarity_dict = {
                    "similarity": api_seq_set_similarity
                }

            else:
                similarity_dict = {"similarity": 0}

        except Exception as e:
            print("error in matrix_api_seq_similarity:", e)
            raise

        return similarity_dict