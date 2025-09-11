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
        >>> ds = DataStatistics()
        >>> ds.mean([1, 2, 3, 4, 5])
        3.00
        """
        if len(data) == 0:
            return 0.00  # Handle empty data case
        return round(sum(data) / len(data), 2)

    def median(self, data):
        """
        Calculate the median of a group of data, accurate to two digits after the Decimal separator
        :param data: list, data list
        :return: float, the median value
        >>> ds = DataStatistics()
        >>> ds.median([2, 5, 1, 3, 4])
        3.00
        """
        n = len(data)
        if n == 0:
            return 0.00  # Handle empty data case
        sorted_data = sorted(data)
        mid = n // 2
        if n % 2 == 0:
            median_value = (sorted_data[mid - 1] + sorted_data[mid]) / 2
        else:
            median_value = sorted_data[mid]
        return round(median_value, 2)

    def mode(self, data):
        """
        Calculate the mode of a set of data
        :param data: list, data list
        :return: list, the mode
        >>> ds = DataStatistics()
        >>> ds.mode([2, 2, 3, 3, 4])
        [2, 3]
        """
        if len(data) == 0:
            return []  # Handle empty data case
        count = Counter(data)
        max_freq = max(count.values())
        modes = [key for key, freq in count.items() if freq == max_freq]
        return modes

# Test cases
if __name__ == "__main__":
    ds = DataStatistics()
    
    # Test case for mean
    mean_output = ds.mean([1, 2, 3, 4, 5])
    print(f"Mean: {mean_output}")  # Expected output: 3.00

    # Test case for median
    median_output = ds.median([2, 5, 1, 3, 4])
    print(f"Median: {median_output}")  # Expected output: 3.00

    # Test case for mode
    mode_output = ds.mode([2, 2, 3, 3, 4])
    print(f"Mode: {mode_output}")  # Expected output: [2, 3]