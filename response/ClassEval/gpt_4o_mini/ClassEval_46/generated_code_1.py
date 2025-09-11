import numpy as np

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
        y_interp = np.interp(x_interp, x, y).tolist()
        return y_interp

    @staticmethod
    def interpolate_2d(x, y, z, x_interp, y_interp):
        """
        Linear interpolation of two-dimensional data
        :param x: The x-coordinate of the data point, list.
        :param y: The y-coordinate of the data point, list.
        :param z: The z-coordinate of the data point, list (2D array).
        :param x_interp: The x-coordinate of the interpolation point, list.
        :param y_interp: The y-coordinate of the interpolation point, list.
        :return: The z-coordinate of the interpolation point, list.
        >>> interpolation = Interpolation()
        >>> interpolation.interpolate_2d([1, 2, 3], [1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1.5, 2.5], [1.5, 2.5])
        [3.0, 7.0]
        """
        z_interp = []
        for xi, yi in zip(x_interp, y_interp):
            z_value = Interpolation._bilinear_interpolate(x, y, z, xi, yi)
            z_interp.append(z_value)
        return z_interp

    @staticmethod
    def _bilinear_interpolate(x, y, z, x_interp, y_interp):
        """
        Helper method for bilinear interpolation.
        """
        x1 = np.searchsorted(x, x_interp) - 1
        x2 = x1 + 1
        y1 = np.searchsorted(y, y_interp) - 1
        y2 = y1 + 1
        
        # Ensure the indices are within bounds
        x1 = max(0, min(x1, len(x)-1))
        x2 = max(0, min(x2, len(x)-1))
        y1 = max(0, min(y1, len(y)-1))
        y2 = max(0, min(y2, len(y)-1))

        # Corner points
        Q11 = z[y1][x1]
        Q12 = z[y2][x1]
        Q21 = z[y1][x2]
        Q22 = z[y2][x2]

        # Interpolation
        if x[x2] != x[x1] and y[y2] != y[y1]:
            return (Q11 * (x2 - x_interp) * (y2 - y_interp) +
                    Q21 * (x_interp - x1) * (y2 - y_interp) +
                    Q12 * (x2 - x_interp) * (y_interp - y1) +
                    Q22 * (x_interp - x1) * (y_interp - y1)) / ((x2 - x1) * (y2 - y1))
        return 0  # If interpolation is not possible

if __name__ == "__main__":
    # Test case for 1D interpolation
    interpolation = Interpolation()
    output_1d = interpolation.interpolate_1d([1, 2, 3], [1, 2, 3], [1.5, 2.5])
    print("1D Interpolation Output:", output_1d)

    # Test case for 2D interpolation
    output_2d = interpolation.interpolate_2d([1, 2, 3], [1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1.5, 2.5], [1.5, 2.5])
    print("2D Interpolation Output:", output_2d)