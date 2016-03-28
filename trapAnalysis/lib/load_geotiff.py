from osgeo import gdal  # For reading tiff files
import numpy as np      # For masked matrices
import util


def load_data_set(filename):
    """
    Load the geotiff data set from the filename
    :param filename: Name of the tiff file
    :return data_set: Geotiff data
    """

    data_set = gdal.Open(filename)

    if data_set is None:
        print "Error retrieving data set"
        return

    return data_set


def get_array_from_band(data_set, band=1):
    """
    Get data from selected band and remove invalid data points
    :param data_set: Geotiff data
    :param band: Different bands hold different information
    :return arr: The array holding the data
    """

    band = data_set.GetRasterBand(band)
    arr = np.ma.masked_array(band.ReadAsArray())

    no_data_value = band.GetNoDataValue()
    if no_data_value:
        arr[arr == no_data_value] = np.ma.masked

    return arr


def set_landscape(data_set):
    """
    Set the metadata for the landscape grid
    :param data_set: Geotiff data
    :return landscape: Object holding metadata for the grid
    """

    geo_transform = data_set.GetGeoTransform()
    nx = data_set.RasterXSize
    ny = data_set.RasterYSize
    landscape = util.Landscape(geo_transform, nx, ny)

    return landscape


def construct_landscape_grid(arr, landscape):
    """
    Create an (m*n)x3 grid holding (x,y,z)-coordinates for all nodes in the grid
    :param arr: Masked array
    :param landscape: Metadata object
    :return:
    """

    unequal_num_of_nodes = (landscape.num_of_nodes_x != landscape.num_of_nodes_y)
    if unequal_num_of_nodes:
        print "The number of nodes in x- and y-direction is not equal"
        return

    # Set x-coordinates, they will be row by row, so the x-grid is repeated
    x_grid = np.arange(landscape.x_min, landscape.x_max, landscape.step_size)
    x = np.tile(x_grid, landscape.num_of_nodes_y)
    landscape.coordinates[:, 0] = x

    # Set y-coordinates, they will be have the same y-value for each row
    y_grid = np.arange(landscape.y_max, landscape.y_min, -landscape.step_size)
    y = np.repeat(y_grid, landscape.num_of_nodes_y)

    landscape.coordinates[:, 1] = y

    # Set z-coordinates
    z = np.reshape(arr, (1, np.product(arr.shape)))[0]
    landscape.coordinates[:, 2] = z
