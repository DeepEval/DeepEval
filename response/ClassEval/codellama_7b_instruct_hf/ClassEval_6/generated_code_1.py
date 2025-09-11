import math

class AvgPartition:
    def __init__(self, lst, limit):
        self.lst = lst
        self.limit = limit

    def setNum(self):
        if self.limit <= 0:
            raise ValueError("Invalid number of partitions")

        self.size = len(self.lst) // self.limit
        self.remainder = len(self.lst) % self.limit

        return (self.size, self.remainder)

    def get(self, index):
        if index >= self.limit:
            raise IndexError("Invalid partition index")

        start = self.size * index
        end = start + self.size

        if index == self.limit - 1:
            end += self.remainder

        return self.lst[start:end]

if __name__ == "__main__":
    lst = list(range(10))
    limit = 3
    partitioner = AvgPartition(lst, limit)
    
    size, remainder = partitioner.setNum()
    print(f"Size of each partition: {size}, Remainder: {remainder}")

    for i in range(limit):
        print(f"Partition {i}: {partitioner.get(i)}")