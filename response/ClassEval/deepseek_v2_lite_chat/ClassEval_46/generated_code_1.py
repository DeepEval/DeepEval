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
        """
        n = len(x)
        if n != len(y) or n != len(x_interp):
            raise ValueError("Input dimensions must match")
        return np.interp(x_interp, x, y)

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
        """
        nx = len(x)
        ny = len(y)
        if nx != ny or nx != len(x_interp) or ny != len(y_interp):
            raise ValueError("Input dimensions must match")
        
        # Convert lists to numpy arrays for efficient interpolation
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        x_interp = np.array(x_interp)
        y_interp = np.array(y_interp)

        # Interpolate along the second dimension
        return np.interp(x_interp, (x, y), (z, z))

# Test cases
if __name__ == "__main__":
    # Test interpolate_1d
    interpolation = Interpolation()
    result1 = interpolation.interpolate_1d([1, 2, 3], [1, 2, 3], [1.5, 2.5])
    print(result1)  # Expected output: [1.5, 2.5]

    # Test interpolate_2d
    result2 = interpolation.interpolate_2d([1, 2, 3], [1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1.5, 2.5], [1.5, 2.5])
    print(result2)  # Expected output: [3.0, 7.0]