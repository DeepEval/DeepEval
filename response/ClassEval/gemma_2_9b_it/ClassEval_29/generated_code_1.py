from collections import Counter
from decimal import Decimal

class DataStatistics:
    """
    This is a class for performing data statistics, supporting to calculate the mean, median, and mode of a given data set.
    """

    def mean(self, data):
        """
        Calculate the average value of a group of data, accurate to two digits after the Decimal separator
        :param data:list, data list
        :return:float, the mean value
        >>> ds = DataStatistics()
        >>> ds.mean([1, 2, 3, 4, 5])
        3.00
        """
        return Decimal(sum(data) / len(data)).quantize(Decimal('0.01'))

    def median(self, data):
        """
        Calculate the median of a group of data, accurate to two digits after the Decimal separator
        :param data:list, data list
        :return:float, the median value
        >>> ds = DataStatistics()
        >>> ds.median([2, 5, 1, 3, 4])
        3.00
        """
        sorted_data = sorted(data)
        data_length = len(data)
        if data_length % 2 == 0:
            return Decimal((sorted_data[data_length // 2 - 1] + sorted_data[data_length // 2]) / 2).quantize(Decimal('0.01'))
        else:
            return Decimal(sorted_data[data_length // 2]).quantize(Decimal('0.01'))

    def mode(self, data):
        """
        Calculate the mode of a set of data
        :param data:list, data list
        :return:float, the mode
        >>> ds = DataStatistics()
        >>> ds.mode([2, 2, 3, 3, 4])
        [2, 3]
        """
        return [item for item, count in Counter(data).items() if count == max(Counter(data).values())]
    
if __name__ == "__main__":
    instance = DataStatistics()
    # Test case for mean
    output_mean = instance.mean([1, 2, 3, 4, 5])
    print(f"Mean: {output_mean}")

    # Test case for median
    output_median = instance.median([2, 5, 1, 3, 4])
    print(f"Median: {output_median}")

    # Test case for mode
    output_mode = instance.mode([2, 2, 3, 3, 4])
    print(f"Mode: {output_mode}")