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
        y_interp = []
        for xi in x_interp:
            i = 0
            while i < len(x) - 1 and x[i] < xi:
                i += 1
            y_interp.append(y[i] + (xi - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]))
        return y_interp

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
        z_interp = []
        for xi, yi in zip(x_interp, y_interp):
            i = 0
            while i < len(x) - 1 and x[i] < xi:
                i += 1
            j = 0
            while j < len(y) - 1 and y[j] < yi:
                j += 1
            z_interp.append(z[i][j] + (xi - x[i]) * (z[i + 1][j] - z[i][j]) / (x[i + 1] - x[i]) + (yi - y[j]) * (z[i][j + 1] - z[i][j]) / (y[j + 1] - y[j]))
        return z_interp

if __name__ == "__main__":
    interpolation = Interpolation()
    # Test case for interpolate_1d
    output_1d = interpolation.interpolate_1d([1, 2, 3], [1, 2, 3], [1.5, 2.5])
    print(f"Output for interpolate_1d: {output_1d}")

    # Test case for interpolate_2d
    output_2d = interpolation.interpolate_2d([1, 2, 3], [1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1.5, 2.5], [1.5, 2.5])
    print(f"Output for interpolate_2d: {output_2d}")