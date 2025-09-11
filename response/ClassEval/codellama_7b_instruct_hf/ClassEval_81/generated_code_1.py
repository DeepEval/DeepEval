import math

class Statistics3:
    def median(self, data):
        """
        calculates the median of the given list.
        :param data: the given list, list.
        :return: the median of the given list, float.
        """
        data.sort()
        if len(data) % 2 == 0:
            median = (data[len(data) // 2 - 1] + data[len(data) // 2]) / 2
        else:
            median = data[len(data) // 2]
        return median

    def mode(self, data):
        """
        calculates the mode of the given list.
        :param data: the given list, list.
        :return: the mode of the given list, list.
        """
        counts = {}
        for item in data:
            if item not in counts:
                counts[item] = 1
            else:
                counts[item] += 1
        max_count = max(counts.values())
        mode = []
        for key, value in counts.items():
            if value == max_count:
                mode.append(key)
        return mode

    def correlation(self, x, y):
        """
        calculates the correlation of the given lists.
        :param x: the first given list, list.
        :param y: the second given list, list.
        :return: the correlation of the given lists, float.
        """
        x_mean = self.mean(x)
        y_mean = self.mean(y)
        x_dev = self.standard_deviation(x)
        y_dev = self.standard_deviation(y)
        cov = 0
        for i in range(len(x)):
            cov += (x[i] - x_mean) * (y[i] - y_mean)
        cov /= len(x) - 1
        return cov / (x_dev * y_dev)

    def mean(self, data):
        """
        calculates the mean of the given list.
        :param data: the given list, list.
        :return: the mean of the given list, float.
        """
        return sum(data) / len(data)

    def correlation_matrix(self, data):
        """
        calculates the correlation matrix of the given list.
        :param data: the given list, list.
        :return: the correlation matrix of the given list, list.
        """
        correlation_matrix = []
        for i in range(len(data)):
            row = []
            for j in range(i + 1, len(data)):
                row.append(self.correlation(data[i], data[j]))
            correlation_matrix.append(row)
        return correlation_matrix

    def standard_deviation(self, data):
        """
        calculates the standard deviation of the given list.
        :param data: the given list, list.
        :return: the standard deviation of the given list, float.
        """
        mean = self.mean(data)
        squared_deviations = []
        for x in data:
            squared_deviations.append((x - mean) ** 2)
        squared_deviations.sort()
        variance = squared_deviations[len(data) // 2]
        return math.sqrt(variance)

    def z_score(self, data):
        """
        calculates the z-score of the given list.
        :param data: the given list, list.
        :return: the z-score of the given list, list.
        """
        mean = self.mean(data)
        standard_deviation = self.standard_deviation(data)
        z_scores = []
        for x in data:
            z_scores.append((x - mean) / standard_deviation)
        return z_scores

if __name__ == "__main__":
    statistics3 = Statistics3()

    # Test case for median
    data = [1, 2, 3, 4, 5]
    output = statistics3.median(data)
    print(output)

    # Test case for mode
    data = [1, 2, 3, 3, 4, 4, 4, 5]
    output = statistics3.mode(data)
    print(output)

    # Test case for correlation
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 4, 5, 6]
    output = statistics3.correlation(x, y)
    print(output)

    # Test case for mean
    data = [1, 2, 3, 4, 5]
    output = statistics3.mean(data)
    print(output)

    # Test case for correlation matrix
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    output = statistics3.correlation_matrix(data)
    print(output)

    # Test case for standard deviation
    data = [1, 2, 3, 4, 5]
    output = statistics3.standard_deviation(data)
    print(output)

    # Test case for z-score
    data = [1, 2, 3, 4, 5]
    output = statistics3.z_score(data)
    print(output)