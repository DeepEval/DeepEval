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
        """
        y_interp = []
        for xp in x_interp:
            for i in range(len(x) - 1):
                if (x[i] <= xp <= x[i + 1]) or (x[i] >= xp >= x[i + 1]):
                    y_interp.append(y[i] + (y[i + 1] - y[i]) * (xp - x[i]) / (x[i + 1] - x[i]))
                    break
            else:
                # If xp is outside the range of x, use the nearest neighbor interpolation
                y_interp.append(y[0] if abs(xp - x[0]) < abs(xp - x[-1]) else y[-1])
        return y_interp

    @staticmethod
    def interpolate_2d(x, y, z, x_interp, y_interp):
        """
        Linear interpolation of two-dimensional data
        :param x: The x-coordinate of the data point, list.
        :param y: The y-coordinate of the data point, list.
        :param z: The z-coordinate of the data point, list of lists.
        :param x_interp: The x-coordinate of the interpolation point, list.
        :param y_interp: The y-coordinate of the interpolation point, list.
        :return: The z-coordinate of the interpolation point, list.
        """
        z_interp = []
        for xp in x_interp:
            for i in range(len(x) - 1):
                if (x[i] <= xp <= x[i + 1]) or (x[i] >= xp >= x[i + 1]):
                    z_interp.append([])
                    for yp in y_interp:
                        for j in range(len(y) - 1):
                            if (y[j] <= yp <= y[j + 1]) or (y[j] >= yp >= y[j + 1]):
                                z_val = (
                                    z[i][j] * (x[i + 1] - xp) * (y[j + 1] - yp) +
                                    z[i + 1][j] * (xp - x[i]) * (y[j + 1] - yp) +
                                    z[i][j + 1] * (x[i + 1] - xp) * (yp - y[j]) +
                                    z[i + 1][j + 1] * (xp - x[i]) * (yp - y[j])
                                ) / ((x[i + 1] - x[i]) * (y[j + 1] - y[j]))
                                z_interp[-1].append(z_val)
                                break
                        else:
                            # If yp is outside the range of y, use the nearest neighbor interpolation
                            z_interp[-1].append(z[i][0] if abs(yp - y[0]) < abs(yp - y[-1]) else z[i][-1])
                    break
            else:
                # If xp is outside the range of x, use the nearest neighbor interpolation
                z_interp.append([])
                for yp in y_interp:
                    z_interp[-1].append(z[0][0] if abs(yp - y[0]) < abs(yp - y[-1]) else z[-1][0])
        return [item for sublist in z_interp for item in sublist]


if __name__ == "__main__":
    interpolation = Interpolation()
    print(interpolation.interpolate_1d([1, 2, 3], [1, 2, 3], [1.5, 2.5]))  # [1.5, 2.5]
    print(interpolation.interpolate_2d([1, 2, 3], [1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1.5, 2.5], [1.5, 2.5]))  # [3.0, 7.0]