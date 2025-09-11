from collections import Counter

class DataStatistics:
    """
    This is a class for performing data statistics, supporting calculation of the mean, median, and mode of a given data set.
    """

    def mean(self, data):
        """
        Calculate the average value of a group of data, accurate to two digits after the decimal separator
        :param data: list, data list
        :return: float, the mean value
        >>> ds = DataStatistics()
        >>> ds.mean([1, 2, 3, 4, 5])
        3.00
        """
        if not data:
            return 0.00
        return round(sum(data) / len(data), 2)

    def median(self, data):
        """
        Calculate the median of a group of data, accurate to two digits after the decimal separator
        :param data: list, data list
        :return: float, the median value
        >>> ds = DataStatistics()
        >>> ds.median([2, 5, 1, 3, 4])
        3.00
        """
        n = len(data)
        if n == 0:
            return 0.00
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
        :return: float, the mode
        >>> ds = DataStatistics()
        >>> ds.mode([2, 2, 3, 3, 4])
        [2, 3]
        """
        if not data:
            return []

        data_counter = Counter(data)
        max_count = max(data_counter.values())
        mode_values = [key for key, count in data_counter.items() if count == max_count]

        return mode_values


if __name__ == "__main__":
    ds = DataStatistics()

    # Test case for mean
    mean_output = ds.mean([1, 2, 3, 4, 5])
    print("Mean:", mean_output)  # Expected output: 3.00

    # Test case for median
    median_output = ds.median([2, 5, 1, 3, 4])
    print("Median:", median_output)  # Expected output: 3.00

    # Test case for mode
    mode_output = ds.mode([2, 2, 3, 3, 4])
    print("Mode:", mode_output)  # Expected output: [2, 3]