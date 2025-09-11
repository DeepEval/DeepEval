from collections import Counter
import statistics
from decimal import Decimal, ROUND_HALF_UP

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
        return Decimal(statistics.mean(data)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def median(self, data):
        """
        Calculate the median of a group of data, accurate to two digits after the Decimal separator
        :param data:list, data list
        :return:float, the median value
        >>> ds = DataStatistics()
        >>> ds.median([2, 5, 1, 3, 4])
        3.00
        """
        return Decimal(statistics.median(data)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def mode(self, data):
        """
        Calculate the mode of a set of data
        :param data:list, data list
        :return:float, the mode
        >>> ds = DataStatistics()
        >>> ds.mode([2, 2, 3, 3, 4])
        [2, 3]
        """
        count = Counter(data)
        max_count = max(count.values())
        modes = [num for num, freq in count.items() if freq == max_count]
        return modes

if __name__ == "__main__":
    instance = DataStatistics()
    
    # Test case for mean
    data = [1, 2, 3, 4, 5]
    output = instance.mean(data)
    print(f"Mean: {output}")
    
    # Test case for median
    data = [2, 5, 1, 3, 4]
    output = instance.median(data)
    print(f"Median: {output}")
    
    # Test case for mode
    data = [2, 2, 3, 3, 4]
    output = instance.mode(data)
    print(f"Mode: {output}")