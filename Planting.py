import numpy as np
from Model import anomaly, cubemodel, subsurface
import copy

class cluster():

    def __init__(self, seed):
        self.seed = copy.copy(seed)
        self.object = [self.seed]
        self.object_num = 1
        self.neighbours = []
        self.neighbours_num = 0
        self.norm = 0
        self.exist = True

class norm():

    def __init__(self, horizontal_x, horizontal_y, depth_z):
        self.dx = abs(horizontal_x[1]-horizontal_x[0])
        self.dy = abs(horizontal_y[1]-horizontal_y[0])
        self.dz = abs(depth_z[1] - depth_z[0])
        self.f = (abs(horizontal_x[-1]-horizontal_x[0]) + abs(horizontal_y[-1]-horizontal_y[0]) + abs(depth_z[-1] - depth_z[0]))/3

class temp_anomaly():

    def __init__(self, measure_square):
        self.neighbours_preanomaly = anomaly(measure_square)
        self.target_preanomaly = []

class target():

    def __init__(self, measure_square):
        self.character = []
        self.loss = 0
        self.gross_err = 0
        self.norm = 0
        self.anomaly = anomaly(measure_square)

class planting():

    def __init__(self, seeds, dg, Gxz, Gyz, Gzz, horizontal_x, horizontal_y, depth_z, error, u):

        self.underground = subsurface(horizontal_x, horizontal_y, depth_z)
        self.measquare = np.meshgrid(horizontal_x, horizontal_x)

        self.seeds = copy.copy(seeds)
        self.seed_num = len(self.seeds)

        self.clusters = {}
        for i in range(self.seed_num):
            self.clusters['cluster' + str(i)] = cluster(self.seeds[i])
        self.cluster = []
        self.cluster = self.clusters['cluster' + str(0)]

        self.anomaly = anomaly(self.measquare)
        self.anomaly.dg = dg
        self.anomaly.Gxz = Gxz
        self.anomaly.Gyz = Gyz
        self.anomaly.Gzz = Gzz

        self.preanomaly = anomaly(self.measquare)
        self.temp_preanomaly = anomaly(self.measquare)

        self.target = target(self.measquare)
        self.norm = norm(horizontal_x, horizontal_y, depth_z)

        self.loss = 0
        self.temp_loss = 0
        self.gross_error = 0
        self.err = error
        self.u = u

        self.initialmodel()
        self.cluster.exist = True

    def initialmodel(self):

        for i in range(self.seed_num):

            x_index, y_index, z_index, property = self.seeds[i][0], self.seeds[i][1], self.seeds[i][2], self.seeds[i][3]
            cube = cubemodel(self.underground.model[x_index, y_index, z_index], property, self.measquare)
            cube.forward()

            self.temp_preanomaly.dg += copy.copy(cube.anomaly.dg)
            self.temp_preanomaly.Gxz += copy.copy(cube.anomaly.Gxz)
            self.temp_preanomaly.Gyz += copy.copy(cube.anomaly.Gyz)
            self.temp_preanomaly.Gzz += copy.copy(cube.anomaly.Gzz)

            self.underground.property[x_index, y_index, z_index] = copy.copy(property)

        self.loss = copy.copy(self.lossfunction())
        self.gross_error = 0
        self.preanomaly = copy.copy(self.temp_preanomaly)

    def searching(self):

        self.cluster.neighbours_num = 0
        self.cluster.neighbours = []

        i = 0

        while True:

            if i == len(self.cluster.object):
                break

            for j in range(3):
                for k in range(2):

                    target = copy.copy(self.cluster.object[i])
                    target[j] += (-1) ** (k + 1)

                    if target[0] < 0:
                        continue
                    if target[0] >= len(self.underground.x_index):
                        continue
                    if target[1] < 0:
                        continue
                    if target[1] >= len(self.underground.y_index):
                        continue
                    if target[2] < 0:
                        continue
                    if target[2] >= len(self.underground.z_index):
                        continue
                    if self.underground.property[target[0], target[1], target[2]] != 0:
                        continue
                    if target in self.cluster.neighbours:
                        continue

                    self.cluster.neighbours_num += 1
                    self.cluster.neighbours.append(copy.copy(target))

            i += 1


    def lossfunction(self):

        loss = np.sum(np.abs(self.anomaly.dg - self.temp_preanomaly.dg)) / np.sum(np.abs(self.anomaly.dg))
        loss += np.sum(np.abs(self.anomaly.Gxz - self.temp_preanomaly.Gxz)) / np.sum(np.abs(self.anomaly.Gxz))
        loss += np.sum(np.abs(self.anomaly.Gyz - self.temp_preanomaly.Gyz)) / np.sum(np.abs(self.anomaly.Gyz))
        loss += np.sum(np.abs(self.anomaly.Gzz - self.temp_preanomaly.Gzz)) / np.sum(np.abs(self.anomaly.Gzz))

        return loss

    def normfunction(self, cube):

            norm = ((cube[0] - self.cluster.seed[0]) * self.norm.dx) ** 2 \
                     + ((cube[1] - self.cluster.seed[1]) * self.norm.dy) ** 2 \
                     + ((cube[2] - self.cluster.seed[2]) * self.norm.dz) ** 2
            norm = np.sqrt(norm)
            norm = 3 * norm / (self.norm.dx + self.norm.dy + self.norm.dz)

            return norm

    def judge(self):

        self.cluster.exist = False
        self.gross_error = 0

        for i in range(self.cluster.neighbours_num):

            x_index, y_index, z_index, property = self.cluster.neighbours[i][0], self.cluster.neighbours[i][1], \
                                                  self.cluster.neighbours[i][2], self.cluster.neighbours[i][3]
            ifcube = cubemodel(self.underground.model[x_index, y_index, z_index], property, self.measquare)
            ifcube.forward()

            self.temp_preanomaly.dg = copy.copy(ifcube.anomaly.dg + self.preanomaly.dg)
            self.temp_preanomaly.Gxz = copy.copy(ifcube.anomaly.Gxz + self.preanomaly.Gxz)
            self.temp_preanomaly.Gyz = copy.copy(ifcube.anomaly.Gyz + self.preanomaly.Gyz)
            self.temp_preanomaly.Gzz = copy.copy(ifcube.anomaly.Gzz + self.preanomaly.Gzz)

            self.target.loss = self.lossfunction()

            if self.target.loss >= self.loss:
                continue

            if abs(self.target.loss - self.loss)/self.loss < self.err:
                continue

            self.target.norm = self.normfunction(self.cluster.neighbours[i]) + self.cluster.norm

            self.target.gross_err = self.target.loss + self.u * self.target.norm

            if self.gross_error == 0:
                self.gross_error = copy.copy(self.target.gross_err)
            if self.target.gross_err > self.gross_error:
                continue

            self.cluster.exist = True
            self.gross_error = copy.copy(self.target.gross_err)
            self.target.character = copy.copy(self.cluster.neighbours[i])
            self.temp_loss = copy.copy(self.target.loss)

            self.target.anomaly = copy.copy(self.temp_preanomaly)

    def update(self):

        if self.cluster.exist == True:

            self.preanomaly = copy.copy(self.target.anomaly)
            self.cluster.object.append(copy.copy(self.target.character))
            self.underground.property[self.target.character[0],self.target.character[1],self.target.character[2]] = self.target.character[3]
            self.loss = copy.copy(self.temp_loss)
            self.cluster.object_num += 1
            self.cluster.norm = copy.copy(self.target.norm)

class iterate(planting):

    def __init__(self, seed, anomaly, Gxz, Gyz, Gzz, horizontal_x, horizontal_y, depth_z, error, u):
        super().__init__(seed, anomaly, Gxz, Gyz, Gzz, horizontal_x, horizontal_y, depth_z, error, u)

        self.error = error
        self.Epoch = 0
        self.end = 0

    def searching(self):
        super(iterate, self).searching()

    def judge(self):
        super(iterate, self).judge()

    def inverison(self):

        while True:

            if self.end == len(self.seeds):
                break

            self.end = 0

            for i in range(len(self.seeds)):

                self.cluster = self.clusters['cluster' + str(i)]

                if self.cluster.exist == False:
                    self.end += 1
                    continue

                self.searching()
                self.judge()
                self.update()

            self.Epoch += 1

            print("Loss = ", self.loss, "Epoch = ", self.Epoch)

        print('no suitable cube')












