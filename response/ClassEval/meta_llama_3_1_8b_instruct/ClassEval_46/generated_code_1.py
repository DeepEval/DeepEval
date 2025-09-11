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
        # Check if x and y have the same length
        if len(x)!= len(y):
            raise ValueError("x and y must have the same length")

        # Check if x_interp is a list
        if not isinstance(x_interp, list):
            raise ValueError("x_interp must be a list")

        # Initialize an empty list to store the interpolated values
        y_interp = []

        # Iterate over each interpolation point
        for x_i in x_interp:
            # Find the indices of the two data points that the interpolation point lies between
            i = 0
            while i < len(x) - 1 and x[i] < x_i:
                i += 1

            # Calculate the interpolated value using linear interpolation
            if i == 0:
                # If the interpolation point lies before the first data point, use the first data point
                y_i = y[0]
            elif i == len(x) - 1:
                # If the interpolation point lies after the last data point, use the last data point
                y_i = y[-1]
            else:
                # Otherwise, use linear interpolation
                y_i = y[i - 1] + (x_i - x[i - 1]) * (y[i] - y[i - 1]) / (x[i] - x[i - 1])

            y_interp.append(y_i)

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
        # Check if x, y, and z have the same length
        if len(x)!= len(y) or len(x)!= len(z) or any(len(row)!= len(x) for row in z):
            raise ValueError("x, y, and z must have the same length")

        # Check if x_interp and y_interp are lists
        if not isinstance(x_interp, list) or not isinstance(y_interp, list):
            raise ValueError("x_interp and y_interp must be lists")

        # Initialize an empty list to store the interpolated values
        z_interp = []

        # Iterate over each interpolation point
        for x_i, y_i in zip(x_interp, y_interp):
            # Find the indices of the two data points that the interpolation point lies between
            i = 0
            while i < len(x) - 1 and x[i] < x_i:
                i += 1

            # If the interpolation point lies outside the data, use the nearest data point
            if i == 0:
                # If the interpolation point lies before the first data point, use the first data point
                z_i = [z[0][j] for j in range(len(z[0]))]
            elif i == len(x) - 1:
                # If the interpolation point lies after the last data point, use the last data point
                z_i = [z[-1][j] for j in range(len(z[0]))]
            else:
                # Otherwise, use linear interpolation in each dimension
                z_i = []
                for j in range(len(z[0])):
                    z_ij = z[i - 1][j] + (x_i - x[i - 1]) * (z[i][j] - z[i - 1][j]) / (x[i] - x[i - 1])
                    z_i.append(z_ij)

            z_interp.append(z_i)

        return z_interp


if __name__ == "__main__":
    interpolation = Interpolation()

    # Test case for interpolate_1d
    x = [1, 2, 3]
    y = [1, 2, 3]
    x_interp = [1.5, 2.5]
    output = interpolation.interpolate_1d(x, y, x_interp)
    print("Test case for interpolate_1d:", output)  # Output: [1.5, 2.5]

    # Test case for interpolate_2d
    x = [1, 2, 3]
    y = [1, 2, 3]
    z = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    x_interp = [1.5, 2.5]
    y_interp = [1.5, 2.5]
    output = interpolation.interpolate_2d(x, y, z, x_interp, y_interp)
    print("Test case for interpolate_2d:", output)  # Output: [[3.0], [7.0]]