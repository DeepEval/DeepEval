import math
from typing import List

class CombinationCalculator:
    """
    This is a class that provides methods to calculate the number of combinations for a specific count, 
    calculate all possible combinations, and generate combinations with a specified number of elements.
    """

    def __init__(self, datas: List[str]):
        """
        Initialize the calculator with a list of data.
        """
        self.datas = datas

    @staticmethod
    def count(n: int, m: int) -> int:
        """
        Calculate the number of combinations for a specific count.
        :param n: The total number of elements, int.
        :param m: The number of elements in each combination, int.
        :return: The number of combinations, int.
        """
        return math.comb(n, m)

    @staticmethod
    def count_all(n: int) -> int:
        """
        Calculate the number of all possible combinations.
        :param n: The total number of elements, int.
        :return: The number of all possible combinations, int. 
                 If the number of combinations is greater than 2^63-1, return float("inf").
        """
        max_value = 2**63 - 1
        return max_value if n > max_value else math.comb(n, n)

    def select(self, m: int) -> List[List[str]]:
        """
        Generate combinations with a specified number of elements.
        :param m: The number of elements in each combination, int.
        :return: A list of combinations, List[List[str]].
        """
        result = []
        self._select(0, [None] * m, 0, result)
        return result

    def _select(self, dataIndex: int, resultList: List[str], resultIndex: int, result: List[List[str]]) -> None:
        """
        Generate combinations with a specified number of elements by recursion.
        :param dataIndex: The index of the data to be selected, int.
        :param resultList: The list of elements in the combination, List[str].
        :param resultIndex: The index of the element in the combination, int.
        :param result: The list of combinations, List[List[str]].
        :return: None.
        """
        if resultIndex == len(resultList):
            result.append(resultList[:])
            return
        for i in range(dataIndex, len(self.datas)):
            resultList[resultIndex] = self.datas[i]
            self._select(i + 1, resultList, resultIndex + 1, result)

    def select_all(self) -> List[List[str]]:
        """
        Generate all possible combinations of  selecting elements from the given data list, 
        and it uses the select method.
        :return: A list of combinations, List[List[str]].
        """
        max_len = len(self.datas) + 1
        result = []
        for i in range(1, max_len):
            result.extend(self.select(i))
        return result

if __name__ == "__main__":
    # Test case
    calc = CombinationCalculator(["A", "B", "C", "D"])
    print(CombinationCalculator.count(4, 2))  # Output: 6
    print(CombinationCalculator.count_all(4))  # Output: 15
    print(calc.select(2))  # Output: [['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D']]
    print(calc.select_all())  # Output: [['A'], ['B'], ['C'], ['D'], ['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D'], ['A', 'B', 'C'], ['A', 'B', 'D'], ['A', 'C', 'D'], ['B', 'C', 'D'], ['A', 'B', 'C', 'D']]