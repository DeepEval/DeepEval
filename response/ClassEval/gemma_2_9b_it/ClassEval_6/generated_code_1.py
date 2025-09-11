class AvgPartition:
    """
    This is a class that partitions the given list into different blocks by specifying the number of partitions, with each block having a uniformly distributed length.
    """

    def __init__(self, lst, limit):
        """
        Initialize the class with the given list and the number of partitions, and check if the number of partitions is greater than 0.
        """
        self.lst = lst
        self.limit = limit
        if self.limit <= 0:
            raise ValueError("Number of partitions must be greater than 0")

    def setNum(self):
        """
        Calculate the size of each block and the remainder of the division.
        :return: the size of each block and the remainder of the division, tuple.
        >>> a = AvgPartition([1, 2, 3, 4], 2)
        >>> a.setNum()
        (2, 0)

        """
        block_size = len(self.lst) // self.limit
        remainder = len(self.lst) % self.limit
        return (block_size, remainder)

    def get(self, index):
        """
        calculate the size of each block and the remainder of the division, and calculate the corresponding start and end positions based on the index of the partition.
        :param index: the index of the partition,int.
        :return: the corresponding block, list.
        >>> a = AvgPartition([1, 2, 3, 4], 2)
        >>> a.get(0)
        [1, 2]

        """
        block_size, remainder = self.setNum()
        start = index * block_size + min(index, remainder)
        end = (index + 1) * block_size + min(index + 1, remainder)
        return self.lst[start:end]


if __name__ == "__main__":
    instance = AvgPartition([1, 2, 3, 4], 2)
    # Test case for setNum
    output = instance.setNum()
    print(f"Output of setNum: {output}")

    # Test case for get
    output = instance.get(0)
    print(f"Output of get(0): {output}")