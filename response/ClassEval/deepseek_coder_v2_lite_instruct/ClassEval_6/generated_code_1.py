class AvgPartition:
    """
    This is a class that partitions the given list into different blocks by specifying the number of partitions, with each block having a uniformly distributed length.
    """

    def __init__(self, lst, limit):
        """
        Initialize the class with the given list and the number of partitions, and check if the number of partitions is greater than 0.
        """
        if limit <= 0:
            raise ValueError("Number of partitions must be greater than 0")
        self.lst = lst
        self.limit = limit

    def setNum(self):
        """
        Calculate the size of each block and the remainder of the division.
        :return: the size of each block and the remainder of the division, tuple.
        """
        total_length = len(self.lst)
        block_size = total_length // self.limit
        remainder = total_length % self.limit
        return (block_size, remainder)

    def get(self, index):
        """
        Calculate the size of each block and the remainder of the division, and calculate the corresponding start and end positions based on the index of the partition.
        :param index: the index of the partition, int.
        :return: the corresponding block, list.
        """
        block_size, remainder = self.setNum()
        if index >= self.limit:
            raise IndexError("Index out of range")
        start = index * block_size + min(index, remainder)
        end = start + block_size + (1 if index < remainder else 0)
        return self.lst[start:end]

# Test cases
if __name__ == "__main__":
    a = AvgPartition([1, 2, 3, 4], 2)
    print(a.setNum())  # Output should be (2, 0)
    print(a.get(0))    # Output should be [1, 2]
    print(a.get(1))    # Output should be [3, 4]

    # Test for index out of range
    try:
        print(a.get(2))
    except IndexError as e:
        print(e)  # Should print "Index out of range"