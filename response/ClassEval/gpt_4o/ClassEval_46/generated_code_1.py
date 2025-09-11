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
        y_interp = np.interp(x_interp, x, y)
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
        >>> interpolation = Interpolation()
        >>> interpolation.interpolate_2d([1, 2, 3], [1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1.5, 2.5], [1.5, 2.5])
        [3.0, 7.0]
        """
        # Convert lists to numpy arrays
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        # Create a meshgrid for the original data
        x_mesh, y_mesh = np.meshgrid(x, y)

        # Flatten the meshgrids and z data for interpolation
        x_flat = x_mesh.flatten()
        y_flat = y_mesh.flatten()
        z_flat = z.flatten()

        # Use griddata to interpolate the data at the given x_interp and y_interp
        points = np.vstack((x_flat, y_flat)).T
        interp_points = np.vstack((x_interp, y_interp)).T
        z_interp = np.array([np.interp(p[0], x, np.interp(p[1], y, z[:, x == p[0]].flatten())) for p in interp_points])
        
        return z_interp.tolist()

if __name__ == "__main__":
    # Test case for 1D interpolation
    instance = Interpolation()
    output_1d = instance.interpolate_1d([1, 2, 3], [1, 2, 3], [1.5, 2.5])
    print("1D Interpolation Output:", output_1d)

    # Test case for 2D interpolation
    output_2d = instance.interpolate_2d([1, 2, 3], [1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1.5, 2.5], [1.5, 2.5])
    print("2D Interpolation Output:", output_2d)