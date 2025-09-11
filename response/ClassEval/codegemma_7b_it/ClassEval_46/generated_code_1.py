# mjt 修改

import bisect 

class Interpolation:
    """
    This is a class that implements the Linear interpolation operation of one-dimensional and two-dimensional data
    """

    def __init__(self):
        pass

    @staticmethod
    def interpolate_1d(x, y, x_interp):
        """
        Linear interpolation of one-dimensional data
        :param x: The x-coordinate of the data point, list.
        :param y: The y-coordinate of the data point, list.
        :param x_interp: The x-coordinate of the interpolation point, list.
        :return: The y-coordinate of the interpolation point, list.
        >>> interpolation = Interpolation()
        >>> interpolation.interpolate_1d([1, 2, 3], [1, 2, 3], [1.5, 2.5])
        [1.5, 2.5]

        """
        output = []
        for x_ in x_interp:
            index = bisect.bisect_left(x, x_)
            if index == 0:
                output.append(y[0])
            elif index == len(x):
                output.append(y[-1])
            else:
                output.append((y[index - 1] + (y[index] - y[index - 1]) / (x[index] - x[index - 1]) * (x_ - x[index - 1])))
        return output

    @staticmethod
    def interpolate_2d(x, y, z, x_interp, y_interp):
        """
        Linear interpolation of two-dimensional data
        :param x: The x-coordinate of the data point, list.
        :param y: The y-coordinate of the data point, list.
        :param z: The z-coordinate of the data point, list.
        :param x_interp: The x-coordinate of the interpolation point, list.
        :param y_interp: The y-coordinate of the interpolation point, list.
        :return: The z-coordinate of the interpolation point, list.
        >>> interpolation = Interpolation()
        >>> interpolation.interpolate_2d([1, 2, 3], [1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1.5, 2.5], [1.5, 2.5])
        [3.0, 7.0]

        """
        output = []
        for x_ in x_interp:
            for y_ in y_interp:
                index_x = bisect.bisect_left(x, x_)
                index_y = bisect.bisect_left(y, y_)
                if index_x == 0 or index_y == 0:
                    output.append(z[0][0])
                elif index_x == len(x) or index_y == len(y):
                    output.append(z[-1][-1])
                else:
                    output.append(z[index_y - 1][index_x - 1] + (z[index_y - 1][index_x] - z[index_y - 1][index_x - 1]) / (x[index_x] - x[index_x - 1]) * (x_ - x[index_x - 1]) +
                                    (z[index_y][index_x - 1] - z[index_y - 1][index_x - 1]) / (y[index_y] - y[index_y - 1]) * (y_ - y[index_y - 1]))
        return output

if __name__ == "__main__":
    interpolation = Interpolation()
    # Test case for interpolate_1d
    output = interpolation.interpolate_1d([1, 2, 3], [1, 2, 3], [1.5, 2.5])
    print(output)
    # Test case for interpolate_2d
    output = interpolation.interpolate_2d([1, 2, 3], [1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1.5, 2.5], [1.5, 2.5])
    print(output)