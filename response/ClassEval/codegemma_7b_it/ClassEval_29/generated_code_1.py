from collections import Counter

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
        if not data:
            raise ValueError("Data list cannot be empty")
        
        sum_of_values = sum(data)
        mean = sum_of_values / len(data)
        return round(mean, 2)

    def median(self, data):
        """
        Calculate the median of a group of data, accurate to two digits after the Decimal separator
        :param data:list, data list
        :return:float, the median value
        >>> ds = DataStatistics()
        >>> ds.median([2, 5, 1, 3, 4])
        3.00
        """
        if not data:
            raise ValueError("Data list cannot be empty")
        
        sorted_data = sorted(data)
        mid = len(sorted_data) // 2
        
        if len(sorted_data) % 2 == 0:
            median = (sorted_data[mid] + sorted_data[mid - 1]) / 2
        else:
            median = sorted_data[mid]
        
        return round(median, 2)

    def mode(self, data):
        """
        Calculate the mode of a set of data
        :param data:list, data list
        :return:float, the mode
        >>> ds = DataStatistics()
        >>> ds.mode([2, 2, 3, 3, 4])
        [2, 3]
        """
        if not data:
            raise ValueError("Data list cannot be empty")
        
        data_counts = Counter(data)
        max_count = max(data_counts.values())
        mode = [num for num, count in data_counts.items() if count == max_count]
        
        return mode
    
if __name__ == "__main__":
    ds = DataStatistics()
    
    # Test case for mean method
    output = ds.mean([1, 2, 3, 4, 5])
    print("Mean:", output)
    
    # Test case for median method
    output = ds.median([2, 5, 1, 3, 4])
    print("Median:", output)
    
    # Test case for mode method
    output = ds.mode([2, 2, 3, 3, 4])
    print("Mode:", output)