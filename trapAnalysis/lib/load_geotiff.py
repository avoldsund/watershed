from osgeo import gdal  # For reading tiff files
import numpy as np      # For masked matrices
import util


def load_geotiff_as_masked_array(filename, band=1):

    data_set = gdal.Open(filename)

    if data_set is None:
        print "Error retrieving data set"
        return

    # Get number of grid points in the plane, as well as their domains
    geo_transform = data_set.GetGeoTransform()
    x_res = data_set.RasterXSize
    y_res = data_set.RasterYSize
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * x_res
    y_min = y_max + geo_transform[5] * y_res

    landscape_metadata = util.LandscapeMetadata(x_res, y_res, x_min, x_max, y_min, y_max)

    # Get data from selected band and remove invalid data points
    band = data_set.GetRasterBand(band)
    arr = np.ma.masked_array(band.ReadAsArray())

    no_data_value = band.GetNoDataValue()
    print no_data_value
    if no_data_value:
        arr[arr == no_data_value] = np.ma.masked

    node_collection = util.NodeCollection(arr)
    print node_collection.coordinates[4000][4000]

    # count number of masked elements in the array
    # print np.ma.count_masked(arr)