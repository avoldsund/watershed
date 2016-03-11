from osgeo import gdal  # For reading tiff files
import numpy as np      # For masked matrices
import util

def load_geotiff_as_masked_array(filename, band = 1):

    data_set = gdal.Open(filename)

    geo_transform = data_set.GetGeoTransform()
    x_res = data_set.RasterXSize
    y_res = data_set.RasterYSize
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * x_res
    y_min = y_max + geo_transform[5] * y_res

    band = data_set.GetRasterBand(band)
    arr = np.ma.masked_array(band.ReadAsArray())

    #Remove no-data values
    no_data_value = band.GetNoDataValue()

    if (no_data_value):
        arr[arr == no_data_value] = np.ma.masked