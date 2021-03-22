import numpy as np
from Model import cubemodel
import matplotlib.pyplot as plt

## forward

x = range(0,2001,50)
y = range(0,2001,50)
z = range(0,501,50)
measquare = np.meshgrid(x,y)

property = 1
location = [300, 700, 1400, 1600, 200, 500]
model = cubemodel(location, property, measquare)
model.forward()

dg = model.anomaly.dg
Gxz = model.anomaly.Gxz
Gyz = model.anomaly.Gyz
Gzz = model.anomaly.Gzz

property = -1
location = [1400, 1600, 400, 600, 200, 400]
model = cubemodel(location,property,measquare)
model.forward()

dg += model.anomaly.dg
Gxz += model.anomaly.Gxz
Gyz += model.anomaly.Gyz
Gzz += model.anomaly.Gzz

fig1 = plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(dg, cmap='jet')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(Gxz, cmap = 'jet')
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(Gyz, cmap = 'jet')
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(Gzz, cmap = 'jet')
plt.colorbar()

## inversion

from Planting import iterate

seeds =list([[10,30,6,1],[30,10,6,-1]])

inversion = iterate(seeds, dg, Gxz, Gyz, Gzz, x, y, z, 1e-3, 10)
inversion.inverison()

fig2 = plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(inversion.preanomaly.dg, cmap='jet')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(inversion.preanomaly.Gxz, cmap = 'jet')
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(inversion.preanomaly.Gyz, cmap = 'jet')
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(inversion.preanomaly.Gzz, cmap = 'jet')
plt.colorbar()

fig3 = plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3, 4, i+1)
    plt.title(str(i*50) + '~' + str((i+1)*50)+'m')
    plt.imshow(inversion.underground.property[:,:,i].T,cmap='bwr')
    plt.colorbar()

plt.show()