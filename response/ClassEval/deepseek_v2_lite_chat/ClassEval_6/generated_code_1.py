class AvgPartition:
    """
    This is a class that partitions the given list into different blocks by specifying the number of partitions, with each block having a uniformly distributed length.
    """

    def __init__(self, lst, limit):
        """
        Initialize the class with the given list and the number of partitions, and check if the number of partitions is greater than 0.
        """
        if limit <= 0:
            raise ValueError("Number of partitions must be greater than zero")
        self.lst = lst
        self.limit = limit
        self.total_length = len(lst)
        self.remainder = self.total_length % limit
        self.block_size = self.total_length // limit
        self.extra = self.total_length - self.block_size * (limit - self.remainder)

    def setNum(self):
        """
        Calculate the size of each block and the remainder of the division.
        :return: the size of each block and the remainder of the division, tuple.
        >>> a = AvgPartition([1, 2, 3, 4], 2)
        >>> a.setNum()
        (2, 0)
        """
        return (self.block_size, self.remainder)

    def get(self, index):
        """
        calculate the size of each block and the remainder of the division, and calculate the corresponding start and end positions based on the index of the partition.
        :param index: the index of the partition,int.
        :return: the corresponding block, list.
        >>> a = AvgPartition([1, 2, 3, 4], 2)
        >>> a.get(0)
        [1, 2]
        """
        start = index * self.block_size
        end = start + self.block_size
        if index < self.remainder:
            end += self.block_size
        return self.lst[start:end]

# Test cases
if __name__ == "__main__":
    import unittest

    class TestAvgPartition(unittest.TestCase):
        def test_setNum(self):
            a = AvgPartition([1, 2, 3, 4], 2)
            self.assertEqual(a.setNum(), (2, 0))

        def test_get(self):
            a = AvgPartition([1, 2, 3, 4], 2)
            self.assertEqual(a.get(0), [1, 2])

    unittest.main()