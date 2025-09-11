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
        >>> a = AvgPartition([1, 2, 3, 4], 2)
        >>> a.setNum()
        (2, 0)
        """
        size = len(self.lst) // self.limit
        remainder = len(self.lst) % self.limit
        return size, remainder

    def get(self, index):
        """
        Calculate the size of each block and the remainder of the division, and calculate the corresponding start and end positions based on the index of the partition.
        :param index: the index of the partition,int.
        :return: the corresponding block, list.
        >>> a = AvgPartition([1, 2, 3, 4], 2)
        >>> a.get(0)
        [1, 2]
        """
        if index < 0 or index >= self.limit:
            raise IndexError("Index out of range")

        size, remainder = self.setNum()
        start = index * size + min(index, remainder)
        end = start + size + (1 if index < remainder else 0)
        return self.lst[start:end]

if __name__ == "__main__":
    # Test case for setNum
    instance = AvgPartition([1, 2, 3, 4], 2)
    output = instance.setNum()
    print(f"setNum() Output: {output}")  # Expected Output: (2, 0)

    # Test case for get
    output = instance.get(0)
    print(f"get(0) Output: {output}")  # Expected Output: [1, 2]

    output = instance.get(1)
    print(f"get(1) Output: {output}")  # Expected Output: [3, 4]