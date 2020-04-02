STATE_SIZE = 3
OBJECT_DIMENSIONALITY = 2
G_MEAN_DIM = (STATE_SIZE, 1)
G_COVARIANCE_DIM = (STATE_SIZE, STATE_SIZE)


class STATE:
    POSITION_X = 0, 0
    POSITION_Y = 1, 0
    VELOCITY = 2, 0
    S_ORIENTATION = 4, 0


class DIMENSION:
    WIDTH = 0
    HEIGHT = 1

# Motion Model
ACCELERATION_REDUCTION_FACTOR = 0.9
RADIAL_VELOCITY_REDUCTION_FACTOR = 0.9
