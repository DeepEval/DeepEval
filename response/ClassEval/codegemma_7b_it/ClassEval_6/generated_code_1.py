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
        self.block_size = len(self.lst) // self.limit
        self.remainder = len(self.lst) % self.limit
        return self.block_size, self.remainder

    def get(self, index):
        """
        calculate the size of each block and the remainder of the division, and calculate the corresponding start and end positions based on the index of the partition.
        :param index: the index of the partition,int.
        :return: the corresponding block, list.
        >>> a = AvgPartition([1, 2, 3, 4], 2)
        >>> a.get(0)
        [1, 2]

        """
        self.block_size, self.remainder = self.setNum()
        if index < 0 or index >= self.limit:
            raise IndexError("Index out of range")
        if index == self.limit - 1 and self.remainder > 0:
            return self.lst[index * self.block_size : (index + 1) * self.block_size + self.remainder]
        else:
            return self.lst[index * self.block_size : (index + 1) * self.block_size]


if __name__ == "__main__":
    # Test case
    a = AvgPartition([1, 2, 3, 4], 2)
    block = a.get(0)
    print(block)