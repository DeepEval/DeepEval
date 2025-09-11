from collections import Counter
import decimal

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
        if not data:
            return 0.00
        total = sum(data)
        mean_value = total / len(data)
        return round(decimal.Decimal(mean_value), 2)

    def median(self, data):
        """
        Calculate the median of a group of data, accurate to two digits after the Decimal separator
        :param data: list, data list
        :return: float, the median value
        """
        if not data:
            return 0.00
        sorted_data = sorted(data)
        n = len(sorted_data)
        mid = n // 2
        if n % 2 == 0:
            median_value = (sorted_data[mid - 1] + sorted_data[mid]) / 2
        else:
            median_value = sorted_data[mid]
        return round(decimal.Decimal(median_value), 2)

    def mode(self, data):
        """
        Calculate the mode of a set of data
        :param data: list, data list
        :return: float, the mode
        """
        if not data:
            return []
        counter = Counter(data)
        max_count = max(counter.values())
        modes = [num for num, count in counter.items() if count == max_count]
        return modes

# Test cases
if __name__ == "__main__":
    ds = DataStatistics()
    
    # Test mean method
    print(ds.mean([1, 2, 3, 4, 5]))  # Expected output: 3.00
    print(ds.mean([10, 20, 30, 40, 50]))  # Expected output: 30.00
    print(ds.mean([]))  # Expected output: 0.00
    
    # Test median method
    print(ds.median([2, 5, 1, 3, 4]))  # Expected output: 3.00
    print(ds.median([1, 3, 3, 4, 5]))  # Expected output: 3.00
    print(ds.median([]))  # Expected output: 0.00
    
    # Test mode method
    print(ds.mode([2, 2, 3, 3, 4]))  # Expected output: [2, 3]
    print(ds.mode([1, 2, 2, 3, 3, 3, 4]))  # Expected output: [3]
    print(ds.mode([]))  # Expected output: []