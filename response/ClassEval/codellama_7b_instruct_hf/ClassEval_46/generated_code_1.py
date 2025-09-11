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
        # Convert the input lists to NumPy arrays
        x = np.array(x)
        y = np.array(y)
        x_interp = np.array(x_interp)

        # Find the indices of the data points that are closest to the interpolation point
        indices = np.searchsorted(x, x_interp)

        # Calculate the interpolation weights for the closest data points
        weights = np.zeros(len(x_interp))
        for i in range(len(x_interp)):
            if indices[i] == len(x):
                weights[i] = (x[indices[i] - 1] - x_interp[i]) / (x[indices[i] - 1] - x[indices[i] - 2])
            else:
                weights[i] = (x_interp[i] - x[indices[i]]) / (x[indices[i] + 1] - x[indices[i]])

        # Calculate the interpolated values using the interpolation weights
        y_interp = np.zeros(len(x_interp))
        for i in range(len(x_interp)):
            y_interp[i] = y[indices[i]] * weights[i] + y[indices[i] - 1] * (1 - weights[i])

        return y_interp.tolist()

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
        # Convert the input lists to NumPy arrays
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        x_interp = np.array(x_interp)
        y_interp = np.array(y_interp)

        # Find the indices of the data points that are closest to the interpolation point
        indices = np.searchsorted(x, x_interp)

        # Calculate the interpolation weights for the closest data points
        weights = np.zeros((len(x_interp), len(y_interp)))
        for i in range(len(x_interp)):
            for j in range(len(y_interp)):
                if indices[i] == len(x):
                    weights[i, j] = (x[indices[i] - 1] - x_interp[i]) / (x[indices[i] - 1] - x[indices[i] - 2])
                else:
                    weights[i, j] = (x_interp[i] - x[indices[i]]) / (x[indices[i] + 1] - x[indices[i]])

        # Calculate the interpolated values using the interpolation weights
        z_interp = np.zeros((len(x_interp), len(y_interp)))
        for i in range(len(x_interp)):
            for j in range(len(y_interp)):
                z_interp[i, j] = z[indices[i], indices[j]] * weights[i, j] + z[indices[i] - 1, indices[j] - 1] * (1 - weights[i, j])

        return z_interp.tolist()
    
if __name__ == "__main__":
    # Example usage of the Interpolation class
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11]
    x_interp = [1.5, 2.5, 3.5]

    interpolator = Interpolation()
    y_interp = interpolator.interpolate_1d(x, y, x_interp)
    print("Interpolated values (1D):", y_interp)

    x_2d = [1, 2, 3]
    y_2d = [1, 2, 3]
    z_2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    x_interp_2d = [1.5, 2.5]
    y_interp_2d = [1.5, 2.5]

    z_interp_2d = interpolator.interpolate_2d(x_2d, y_2d, z_2d, x_interp_2d, y_interp_2d)
    print("Interpolated values (2D):", z_interp_2d)