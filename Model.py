import numpy as np
from numpy import log, arctan, square, sqrt

G = 6.67259e-11

class anomaly():

    def __init__(self, measure_square):
        self.dg = np.zeros(measure_square[0].shape)
        self.Gxz = np.zeros(measure_square[0].shape)
        self.Gyz = np.zeros(measure_square[0].shape)
        self.Gzz = np.zeros(measure_square[0].shape)

class cubemodel():

    def __init__(self, location, property, measure_square):
        self.property = property
        self.location = location
        self.measure_square = measure_square
        self.anomaly = anomaly(measure_square)

    def forward(self):
        # measure square
        X, Y = self.measure_square[0], self.measure_square[1]
        # location of model
        x = self.location[0:2]
        y = self.location[2:4]
        z = self.location[4:6]

        # compute anomaly from single cube
        for i in range(2):
            for j in range(2):
                for k in range(2):

                    xd = X - x[i]
                    yd = Y - y[j]
                    r = square(xd) + square(yd) + square(z[k]+0.1)
                    r = sqrt(r)

                    weight = (-1)**(i+j+k+1)
                    self.anomaly.dg +=  weight * ( - xd*log(r+yd) - yd*log(r+xd)- (z[k]+0.1)*arctan(-xd*yd/r/(z[k]+0.1)) )
                    self.anomaly.Gzz += weight * ( - arctan( xd*yd / (z[k]+0.1) / r ) )
                    self.anomaly.Gxz -= weight * ( log(r+yd) )
                    self.anomaly.Gyz -= weight * ( log(r+xd) )

        self.anomaly.dg *= self.property*G*1e8 # mGal
        self.anomaly.Gzz *= self.property*G*1e12 # E
        self.anomaly.Gxz *= self.property*G*1e12
        self.anomaly.Gyz *= self.property*G*1e12

class subsurface:

    def __init__(self, horizontal_x, horizontal_y, depth_z):
        self.x_index = [i for i in range(len(horizontal_x) - 1)]
        self.y_index = [i for i in range(len(horizontal_y) - 1)]
        self.z_index = [i for i in range(len(depth_z) - 1)]
        self.model = np.zeros( (len(self.x_index ), len(self.y_index), len(self.z_index), 7) )
        self.property = np.zeros( (len(self.x_index ), len(self.y_index), len(self.z_index)) )
        self.character = np.zeros( (len(self.x_index ), len(self.y_index), len(self.z_index)) )

        for i in self.x_index:
            for j in self.y_index:
                for k in self.z_index:
                    self.model[i, j, k, 0] = horizontal_x[i]
                    self.model[i, j, k, 1] = horizontal_x[i+1]
                    self.model[i, j, k, 2] = horizontal_y[j]
                    self.model[i, j, k, 3] = horizontal_y[j+1]
                    self.model[i, j, k, 4] = depth_z[k]
                    self.model[i, j, k, 5] = depth_z[k+1]


