class AvgPartition:
    """
    This is a class that partitions the given list into different blocks by specifying the number of partitions, 
    with each block having a uniformly distributed length.
    """

    def __init__(self, lst, limit):
        """
        Initialize the class with the given list and the number of partitions, and check if the number of partitions is greater than 0.
        """
        if limit <= 0:
            raise ValueError("Number of partitions should be greater than 0")

        self.lst = lst
        self.limit = limit

    def setNum(self):
        """
        Calculate the size of each block and the remainder of the division.
        :return: the size of each block and the remainder of the division, tuple.
        >>> a = AvgPartition([1, 2, 3, 4], 2)
        >>> a.setNum()
        (2, 0)
        """
        block_size, remainder = divmod(len(self.lst), self.limit)
        return block_size, remainder

    def get(self, index):
        """
        Calculate the size of each block and the remainder of the division, 
        and calculate the corresponding start and end positions based on the index of the partition.
        :param index: the index of the partition,int.
        :return: the corresponding block, list.
        >>> a = AvgPartition([1, 2, 3, 4], 2)
        >>> a.get(0)
        [1, 2]
        """
        block_size, remainder = self.setNum()
        start = index * (block_size + (1 if remainder > index else 0))
        end = start + block_size

        return self.lst[start:end]


if __name__ == "__main__":
    a = AvgPartition([1, 2, 3, 4, 5, 6], 2)
    print(a.setNum())  # Output: (3, 0)
    print(a.get(0))    # Output: [1, 2, 3]
    print(a.get(1))    # Output: [4, 5, 6]

    a = AvgPartition([1, 2, 3, 4, 5, 6], 3)
    print(a.setNum())  # Output: (2, 0)
    print(a.get(0))    # Output: [1, 2]
    print(a.get(1))    # Output: [3, 4]
    print(a.get(2))    # Output: [5, 6]

    a = AvgPartition([1, 2, 3, 4, 5, 6], 6)
    print(a.setNum())  # Output: (1, 0)
    print(a.get(0))    # Output: [1]
    print(a.get(1))    # Output: [2]
    print(a.get(2))    # Output: [3]
    print(a.get(3))    # Output: [4]
    print(a.get(4))    # Output: [5]
    print(a.get(5))    # Output: [6]