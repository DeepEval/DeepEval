from statistics import mean, median, mode
from collections import Counter

class DataStatistics:
    """
    This is a class for performing data statistics, supporting to calculate the mean, median, and mode of a given data set.
    """

    def mean(self, data):
        """
        Calculate the average value of a group of data, accurate to two digits after the Decimal separator
        :param data: list, data list
        :return: float, the mean value
        """
        return round(mean(data), 2)

    def median(self, data):
        """
        Calculate the median of a group of data, accurate to two digits after the Decimal separator
        :param data: list, data list
        :return: float, the median value
        """
        return round(median(data), 2)

    def mode(self, data):
        """
        Calculate the mode of a set of data
        :param data: list, data list
        :return: list, the mode
        """
        data_dict = dict(Counter(data))
        modes = [k for k, v in data_dict.items() if v == max(list(data_dict.values()))]
        return modes

# Test cases
if __name__ == "__main__":
    ds = DataStatistics()

    # Test case for mean
    mean_output = ds.mean([1, 2, 3, 4, 5])
    print(f"Mean of [1, 2, 3, 4, 5]: {mean_output}")

    # Test case for median
    median_output = ds.median([2, 5, 1, 3, 4])
    print(f"Median of [2, 5, 1, 3, 4]: {median_output}")

    # Test case for mode
    mode_output = ds.mode([2, 2, 3, 3, 4])
    print(f"Mode of [2, 2, 3, 3, 4]: {mode_output}")