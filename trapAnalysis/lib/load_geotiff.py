from osgeo import gdal  # For reading tiff files
import numpy as np      # For masked matrices
import util
import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
import trap_analysis
import plot


class DataFileMetadata:

    def __init__(self, ds):

        geo_transform = ds.GetGeoTransform()
        self.nx = ds.RasterXSize
        self.ny = ds.RasterYSize
        self.boolean_arr = get_array_from_band(ds, band=1)
        self.x_min = geo_transform[0]
        self.y_max = geo_transform[3]
        self.x_max = self.x_min + geo_transform[1] * (self.nx - 1)
        self.y_min = self.y_max + geo_transform[5] * (self.ny - 1)
        self.total_number_of_nodes = self.nx * self.ny
        step_size_x = geo_transform[1]
        step_size_y = geo_transform[5]
        unequal_step_size = (abs(step_size_x) != abs(step_size_y))
        if unequal_step_size:
            print 'The step size in the x- and y-direction is not equal.'
            print 'Make sure that your data is a cartesian grid with points spaced 10 meters apart.'
            return
        self.step_size = step_size_x


def load_ds(filename):
    """
    Load the geotiff data set from the filename
    :param filename: Name of the tiff file
    :return ds: Geotiff data
    """

    ds = gdal.Open(filename)

    if ds is None:
        print "Error retrieving data set"
        return

    return ds


def get_array_from_band(ds, band=1):
    """
    Get data from selected band and remove invalid data points
    :param ds: Geotiff data
    :param band: Different bands hold different information
    :return arr: The array holding the data
    """

    band = ds.GetRasterBand(band)
    arr = np.ma.masked_array(band.ReadAsArray())
    no_data_value = band.GetNoDataValue()

    if no_data_value:
        arr[arr == no_data_value] = np.ma.masked

    return arr


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
    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.num_of_nodes_x)
    x = np.tile(x_grid, landscape.num_of_nodes_y)
    landscape.coordinates[:, 0] = x

    # Set y-coordinates, they will be have the same y-value for each row
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.num_of_nodes_y)
    y = np.repeat(y_grid, landscape.num_of_nodes_y)

    landscape.coordinates[:, 1] = y

    # Set z-coordinates
    z = np.reshape(arr, (1, np.product(arr.shape)))[0]
    landscape.coordinates[:, 2] = z

    landscape.arr = arr


def get_landscape(file_name):
    """
    Returns a landscape object given a tiff file. This object contains all coordinates, downslope neighbors etc.
    :param file_name: File name of the .tiff data file
    :return landscape: Landscape object
    """

    ds = load_ds(file_name)
    arr = get_array_from_band(ds)
    landscape = trap_analysis.Landscape(ds)
    construct_landscape_grid(arr, landscape)

    return landscape


def get_landscape_tyrifjorden(file_name):
    """
    Return a modified landscape object because the Tyrifjorden area have NaN values for first column and last row
    :param file_name:
    :return:
    """

    landscape = get_landscape(file_name)
    modify_landscape_tyrifjorden(landscape)

    return landscape


def modify_landscape_tyrifjorden(landscape): # Specific changes _ONLY_ for Tyrifjorden, 1. col, last row NaN
    """
    Modification necessary for removing the first column and the last row
    :param landscape: The landscape object to be modified
    :return:
    """

    landscape.x_min = landscape.x_min + landscape.step_size
    landscape.y_min = landscape.y_min + landscape.step_size
    landscape.num_of_nodes_x -= 1
    landscape.num_of_nodes_y -= 1
    landscape.total_number_of_nodes = landscape.num_of_nodes_x * landscape.num_of_nodes_y
    landscape.arr = landscape.arr[:-1, 1:]
    landscape.coordinates = np.empty((landscape.total_number_of_nodes, 3))

    # Set x-coordinates, they will be row by row, so the x-grid is repeated
    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.num_of_nodes_x)
    x = np.tile(x_grid, landscape.num_of_nodes_y)
    landscape.coordinates[:, 0] = x

    # Set y-coordinates, they will be have the same y-value for each row
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.num_of_nodes_y)
    y = np.repeat(y_grid, landscape.num_of_nodes_y)

    landscape.coordinates[:, 1] = y

    # Set z-coordinates
    z = np.reshape(landscape.arr, (1, np.product(landscape.arr.shape)))[0]
    landscape.coordinates[:, 2] = z


def get_lake_river_marsh_information_tyrifjorden(landscape, lake_file, river_file, small_rivers_file, marsh_file):
    """
    Return a 2d boolean array indicating whether or not there is a lake, a river or a marsh.
    :param landscape: Landscape object.
    :param lake_file: Data file for lake.
    :param river_file: Data file for river.
    :param small_rivers_file: Data file for small rivers.
    :param marsh_file: Data file for marshes.
    :return lakes_rivers_marshes: Boolean array saying if there is a lake, a marsh or river.
    """

    lakes = fit_data_in_landscape_tyrifjorden(landscape, lake_file)
    rivers = fit_data_in_landscape_tyrifjorden(landscape, river_file)
    small_rivers = fit_data_in_landscape_tyrifjorden(landscape, small_rivers_file)
    marshes = fit_data_in_landscape_tyrifjorden(landscape, marsh_file)

    # lakes_rivers_marshes = lakes.astype(bool) + rivers.astype(bool) + small_rivers.astype(bool) + marshes.astype(bool)
    print lakes
    print rivers
    print small_rivers
    print marshes
    # print lakes_rivers_marshes
    # print 'Total: ', np.sum(lakes_rivers_marshes)
    # print 'Lake: ', np.sum(lakes)
    # print 'River: ', np.sum(rivers)
    # print 'Small rivers: ', np.sum(small_rivers)
    # print 'Marshes: ', np.sum(marshes)

    return lakes, rivers, small_rivers, marshes


def fit_data_in_landscape_tyrifjorden(landscape, data_file):
    """
    Return the boolean array for the data file, e.g. rivers, lakes etc.
    :param landscape: The landscape object.
    :param data_file: The data file with information about lakes or rivers etc.
    :return new_boolean_arr: The new boolean array where the invalid column and row is removed.
    """

    ds = load_ds(data_file)
    info = DataFileMetadata(ds)

    # Remove first column no matter the dimensions
    info.nx -= 1
    info.boolean_arr = info.boolean_arr[:, 1:]  # Remove first column

    new_boolean_arr = np.zeros((landscape.num_of_nodes_y, landscape.num_of_nodes_x), dtype=int)
    range_x = info.nx if (landscape.num_of_nodes_x >= info.nx) else landscape.num_of_nodes_x
    range_y = info.ny if (landscape.num_of_nodes_y >= info.ny) else landscape.num_of_nodes_y
    info.ny = range_y
    info.nx = range_x
    new_boolean_arr[0:range_y, 0:range_x] = info.boolean_arr[0:range_y, 0:range_x]
    # This will automatically remove the last row if there are
    # more rows than in the landscape. If there are less rows, nothing will be affected.
    info.boolean_arr = new_boolean_arr

    return new_boolean_arr


def get_lake_river_marsh_information(landscape, lake_file, river_file, small_rivers_file, marsh_file):
    """
    Return a 2d boolean array indicating whether or not there is a lake, a river or a marsh.
    :param landscape: Landscape object.
    :param lake_file: Data file for lake.
    :param river_file: Data file for river.
    :param small_rivers_file: Data file for small rivers.
    :param marsh_file: Data file for marshes.
    :return lakes_rivers_marshes: Boolean array saying if there is a lake, a marsh or river.
    """

    lakes = fit_data_in_landscape(landscape, lake_file)
    rivers = fit_data_in_landscape(landscape, river_file)
    small_rivers = fit_data_in_landscape(landscape, small_rivers_file)
    marshes = fit_data_in_landscape(landscape, marsh_file)

    # There are overlap between lakes, rivers, small rivers and marshes. Good to know!
    # print 'Lakes, rivers: ', len(np.where(lakes[np.where(lakes == rivers)[0]] == 1)[0])
    # print 'Lakes, small_rivers: ', len(np.where(lakes[np.where(lakes == small_rivers)[0]] == 1)[0])
    # print 'Lakes, marshes: ', len(np.where(lakes[np.where(lakes == marshes)[0]] == 1)[0])

    return lakes, rivers, small_rivers, marshes


def fit_data_in_landscape(landscape, data_file):
    """
    If the additional data, such as lakes or rivers, have a larger grid than the landscape, only the part covering the
    landscape is used. If the grid is smaller, only the information in the whole grid is used, the rest of the landscape
    will not get any new information.
    :param landscape: The landscape object.
    :param data_file: The information, such as data file over rivers, marshes etc.
    :return new_data: 2d-grid with 1 or 0 for different fields. 1 if there is a river e.g, 0 if not.
    """

    data_set = load_ds(data_file)
    info = DataFileMetadata(data_set)

    # To make it less general, we assume that x_min and y_max for the landscape and the data file
    # will agree with each other, which, hopefully, they always do.
    if (landscape.x_min != info.x_min) or (landscape.y_max != info.y_max):
        print 'The x-min- or y-max-coordinates do not agree for the data file and the landscape. Aborting.'
        return

    new_data = np.zeros((landscape.num_of_nodes_y, landscape.num_of_nodes_x), dtype=int)
    range_x = info.nx if (landscape.num_of_nodes_x >= info.nx) else landscape.num_of_nodes_x
    range_y = info.ny if (landscape.num_of_nodes_y >= info.ny) else landscape.num_of_nodes_y
    new_data[0:range_y, 0:range_x] = info.boolean_arr[0:range_y, 0:range_x]

    return new_data
