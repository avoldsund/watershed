from numpy import matrix

class NodeCollection:

    coordinates = None

    def __init__(self, coordinates):
        self.coordinates = coordinates








class LandscapeMetadata:
    num_of_cells_x = None
    num_of_cells_y = None
    x_coord_min = None
    x_coord_max = None
    y_coord_min = None
    y_coord_max = None

    def __init__(self, num_of_cells_x, num_of_cells_y, x_coord_min, x_coord_max, y_coord_min, y_coord_max):
        self.num_of_cells_x = num_of_cells_x
        self.num_of_cells_y = num_of_cells_y
        self.x_coord_min = x_coord_min
        self.x_coord_max = x_coord_max
        self.y_coord_min = y_coord_min
        self.y_coord_max = y_coord_max