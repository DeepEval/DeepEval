class AvgPartition:
    """
    This is a class that partitions the given list into different blocks by specifying the number of partitions,
    with each block having a uniformly distributed length.
    """

    def __init__(self, lst, limit):
        """
        Initialize the class with the given list and the number of partitions, and check if the number of 
        partitions is greater than 0.
        """
        self.lst = lst
        self.limit = limit
        if limit <= 0:
            raise ValueError("Number of partitions must be greater than 0.")

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
        return block_size, remainder

    def get(self, index):
        """
        Calculate the size of each block and the remainder of the division, and calculate the corresponding 
        start and end positions based on the index of the partition.
        :param index: the index of the partition, int.
        :return: the corresponding block, list.
        >>> a = AvgPartition([1, 2, 3, 4], 2)
        >>> a.get(0)
        [1, 2]
        >>> a.get(1)
        [3, 4]
        """
        block_size, remainder = self.setNum()
        start = index * block_size + min(index, remainder)
        end = start + block_size + (1 if index < remainder else 0)
        return self.lst[start:end]

if __name__ == "__main__":
    # Test case for setNum
    instance_set_num = AvgPartition([1, 2, 3, 4], 2)
    output_set_num = instance_set_num.setNum()
    print(output_set_num)  # Expected output: (2, 0)

    # Test cases for get
    instance_get = AvgPartition([1, 2, 3, 4, 5], 3)
    output_get_0 = instance_get.get(0)
    output_get_1 = instance_get.get(1)
    output_get_2 = instance_get.get(2)
    
    print(output_get_0)  # Expected output: [1, 2]
    print(output_get_1)  # Expected output: [3, 4]
    print(output_get_2)  # Expected output: [5]