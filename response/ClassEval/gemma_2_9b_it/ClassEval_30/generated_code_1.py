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
        return np.sum(self.data)

    def get_min(self):
        """
        Calculate the minimum value in the data
        :return:float
        >>> ds2 = DataStatistics2([1, 2, 3, 4])
        >>> ds2.get_min()
        1
        """
        return np.min(self.data)

    def get_max(self):
        """
        Calculate the maximum value in the data
        :return:float
        >>> ds2 = DataStatistics2([1, 2, 3, 4])
        >>> ds2.get_max()
        4
        """
        return np.max(self.data)

    def get_variance(self):
        """
        Calculate variance, accurate to two digits after the Decimal separator
        :return:float
        >>> ds2 = DataStatistics2([1, 2, 3, 4])
        >>> ds2.get_variance()
        1.25
        """
        return round(np.var(self.data), 2)

    def get_std_deviation(self):
        """
        Calculate standard deviation, accurate to two digits after the Decimal separator
        :return:float
        >>> ds2 = DataStatistics2([1, 2, 3, 4])
        >>> ds2.get_std_deviation()
        1.12
        """
        return round(np.std(self.data), 2)

    def get_correlation(self):
        """
        Calculate correlation
        :return:float
        >>> ds2 = DataStatistics2([1, 2, 3, 4])
        >>> ds2.get_correlation()
        1.0
        """
        # Assuming you want to calculate correlation with itself
        return np.corrcoef(self.data, self.data)[0, 1] 
    
if __name__ == "__main__":
    instance = DataStatistics2([1, 2, 3, 4])
    # Test case for get_sum
    output = instance.get_sum()
    print(f"Sum: {output}")

    # Test case for get_min
    output = instance.get_min()
    print(f"Minimum: {output}")

    # Test case for get_max
    output = instance.get_max()
    print(f"Maximum: {output}")

    # Test case for get_variance
    output = instance.get_variance()
    print(f"Variance: {output}")

    # Test case for get_std_deviation
    output = instance.get_std_deviation()
    print(f"Standard Deviation: {output}")

    # Test case for get_correlation
    output = instance.get_correlation()
    print(f"Correlation: {output}")