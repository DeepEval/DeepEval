import numpy as np

class DataStatistics2:
    """
    This is a class for performing data statistics, supporting to get the sum, minimum, maximum, variance, standard deviation, and correlation of a given dataset.
    """

    def __init__(self, data):
        """
        Initialize Data List
        :param data:list
        """
        self.data = np.array(data)

    def get_sum(self):
        """
        Calculate the sum of data
        :return:float
        >>> ds2 = DataStatistics2([1, 2, 3, 4])
        >>> ds2.get_sum()
        10
        """
        return float(np.sum(self.data))

    def get_min(self):
        """
        Calculate the minimum value in the data
        :return:float
        >>> ds2 = DataStatistics2([1, 2, 3, 4])
        >>> ds2.get_min()
        1
        """
        return float(np.min(self.data))

    def get_max(self):
        """
        Calculate the maximum value in the data
        :return:float
        >>> ds2 = DataStatistics2([1, 2, 3, 4])
        >>> ds2.get_max()
        4
        """
        return float(np.max(self.data))

    def get_variance(self):
        """
        Calculate variance, accurate to two digits after the Decimal separator
        :return:float
        >>> ds2 = DataStatistics2([1, 2, 3, 4])
        >>> ds2.get_variance()
        1.25
        """
        return round(float(np.var(self.data, ddof=0)), 2)

    def get_std_deviation(self):
        """
        Calculate standard deviation, accurate to two digits after the Decimal separator
        :return:float
        >>> ds2 = DataStatistics2([1, 2, 3, 4])
        >>> ds2.get_std_deviation()
        1.12
        """
        return round(float(np.std(self.data, ddof=0)), 2)

    def get_correlation(self):
        """
        Calculate correlation. Here we'll assume a simple case where we correlate the data with itself.
        :return:float
        >>> ds2 = DataStatistics2([1, 2, 3, 4])
        >>> ds2.get_correlation()
        1.0
        """
        if len(self.data) < 2:
            return None  # Correlation is not defined for fewer than 2 points
        return float(np.corrcoef(self.data, self.data)[0, 1])

if __name__ == "__main__":
    # Test cases
    ds2 = DataStatistics2([1, 2, 3, 4])
    
    print("Sum:", ds2.get_sum())                   # Output: 10.0
    print("Min:", ds2.get_min())                   # Output: 1.0
    print("Max:", ds2.get_max())                   # Output: 4.0
    print("Variance:", ds2.get_variance())         # Output: 1.25
    print("Standard Deviation:", ds2.get_std_deviation())  # Output: 1.12
    print("Correlation:", ds2.get_correlation())   # Output: 1.0