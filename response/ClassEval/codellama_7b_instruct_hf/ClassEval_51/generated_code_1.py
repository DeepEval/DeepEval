import numpy as np
import statsmodels.stats.multicomp as sm

class KappaCalculator:
    def kappa(self, testData, k):
        # Cohen's kappa calculation
        # testData is a k-dimensional matrix
        # k is the dimension of the matrix
        n = testData.shape[0]
        m = testData.shape[1]
        p = (m - 1) / (n - 1)
        q = (m - 1) / (n - 1)
        s = np.sum(testData, axis=0)
        c = np.sum(s) / n
        d = np.sum((s - c) ** 2) / (n - 1)
        return (p - q) / (p + q)

    def fleiss_kappa(self, testData, N, k, n):
        # Fleiss' kappa calculation
        # testData is an N * k matrix
        # N is the number of samples
        # k is the number of categories
        # n is the number of raters
        s = np.sum(testData, axis=0)
        c = np.sum(s) / N
        d = np.sum((s - c) ** 2) / (N - 1)
        p = (k - 1) / (k + 1)
        q = (k - 1) / (k + 1)
        return (p - q) / (p + q)
    
if __name__ == "__main__":
    # Example usage
    testData = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    kappa_calculator = KappaCalculator()
    
    # Cohen's kappa
    kappa_value = kappa_calculator.kappa(testData, 3)
    print(f"Cohen's kappa: {kappa_value}")

    # Fleiss' kappa
    fleiss_kappa_value = kappa_calculator.fleiss_kappa(testData, 3, 3, 3)
    print(f"Fleiss' kappa: {fleiss_kappa_value}")