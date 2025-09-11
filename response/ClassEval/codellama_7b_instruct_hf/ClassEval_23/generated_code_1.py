import math
from typing import List

class CombinationCalculator:
    def __init__(self, datas: List[str]):
        self.datas = datas

    @staticmethod
    def count(n: int, m: int) -> int:
        return math.factorial(n) // (math.factorial(m) * math.factorial(n - m))

    @staticmethod
    def count_all(n: int) -> int:
        return math.factorial(n)

    def select(self, m: int) -> List[List[str]]:
        result = []
        for i in range(len(self.datas)):
            for j in range(i+1, len(self.datas)):
                result.append([self.datas[i], self.datas[j]])
        return result

    def select_all(self) -> List[List[str]]:
        result = []
        for i in range(len(self.datas)):
            result.append([self.datas[i]])
            for j in range(i+1, len(self.datas)):
                result.append([self.datas[i], self.datas[j]])
        return result

    def _select(self, dataIndex: int, resultList: List[str], resultIndex: int, result: List[List[str]]):
        if resultIndex == len(resultList):
            result.append(resultList[:])
            return
        for i in range(dataIndex, len(self.datas)):
            resultList[resultIndex] = self.datas[i]
            self._select(i+1, resultList, resultIndex+1, result)

if __name__ == "__main__":
    calc = CombinationCalculator(["A", "B", "C", "D"])
    output = calc.count(4, 2)
    print(output)  # 6

    output = calc.count_all(4)
    print(output)  # 15

    output = calc.select(2)
    print(output)  # [['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D']]

    output = calc.select_all()
    print(output)  # [['A'], ['B'], ['C'], ['D'], ['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D'], ['A', 'B', 'C'], ['A', 'B', 'D'], ['A', 'C', 'D'], ['B', 'C', 'D'], ['A', 'B', 'C', 'D']]