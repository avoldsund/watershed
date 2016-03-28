from osgeo import gdal  # For reading tiff files
import numpy as np      # For masked matrices
import util


def load_dataset(filename):

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
    :return: The array holding the data
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
    :return:
    """

    geo_transform = data_set.GetGeoTransform()
    nx = data_set.RasterXSize
    ny = data_set.RasterYSize
    landscape = util.Landscape(geo_transform, nx, ny)

    return landscape


def construct_landscape_grid(data_set, landscape):

    unequal_num_of_nodes = (landscape.num_of_nodes_x != landscape.num_of_nodes_y)
    if unequal_num_of_nodes:
        print "The number of nodes in x- and y-direction is not equal"
        return

    # Set x-coordinates, they will be row by row, so the x-grid is repeated
    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.num_of_nodes_x)
    x = np.tile(x_grid, landscape.num_of_nodes_y)
    landscape.coordinates[:, 0] = x

    # Set y-coordinates, they will be have the same y-value for each row
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.num_of_nodes_y)
    y = np.repeat(y_grid, landscape.num_of_nodes_y)
    landscape.coordinates[:, 1] = y

    # Set z-coordinates
    arr_in_2d = get_array_from_band(data_set)
    z = np.flatten(arr_in_2d) # This is possible to do faster, just wanted a working version
    landscape.coordinates[:, 2] = z