import numpy as np
from polykriging import curve2D, utility, tool
import sympy as sym
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema  # data compression

# the current work directory
cwd = utility.cwd_chdir()

# Load centroid data and assigned to the bezier points
filepath = "D:/04_coding/Python/06_Visualization/Mesh/00_data_acquisition/3D_2_4_5layers/Long/centerline/0.csv"
# filepath = "D:/04_coding/Python/06_Visualization/Mesh/00_data_acquisition/3D_2_4_5layers" \
#            "/Binder/Centerline/uBinder2/centerline.csv"
centerline = np.loadtxt(filepath, delimiter = ',', skiprows = 1)


# Reorganize the data for parametric kriging:
# x
matXC = centerline[:, [0,2]]
matXC[:, [0,1]] = matXC[:, [1,0]]

matYC = centerline[:, [1,2]]
matYC[:, [0,1]] = matYC[:, [1,0]]

# for local maxima
max_ind = argrelextrema(matYC[:,1], np.greater_equal)
# for local minima
min_ind = argrelextrema(matYC[:,1], np.less)
extrema_ind = np.hstack((max_ind, min_ind))

add = [  [2,  3,  4,  6,  9, 52, 53, 58, 60, 64, 73, 75,
 76, 77, 78, 97, 99, 120, 121, 123, 128, 157, 159, 161,
 162, 171]]

extrema_ind = np.hstack((extrema_ind, add))

# include end points
extrema_ind = np.insert(extrema_ind,0,[0],axis=1)
extrema_ind = np.insert(extrema_ind,-1,[len(matYC[:,1])-1],axis=1)
# avoid repeat
extrema_ind = np.unique(extrema_ind)
extrema_ind.sort()   # sort by ascending

xmat_krig, xmat_krig_inv, xvector_ba, xexpr = curve2D.curveKrig(
    matXC[extrema_ind], 'const', 'lin',nuggetEffect=5e-2)
ymat_krig, ymat_krig_inv, yvector_ba, yexpr = curve2D.curveKrig(
    matYC[extrema_ind], 'quad', 'lin',nuggetEffect=5e-2)


##expr = {'Weft':{"1":[xexpr, yexpr]}}
### Save the kriging expression
##utility.expr_io("centerlineKrig.txt", expr)

#---------------------------------------------------------------------
# Calculate the Kriged coordinate and the tangent
x = sym.symbols('x')

coorKrig = np.zeros(centerline.shape)
xf = sym.lambdify(x, xexpr, 'numpy')
coorKrig[:,0] = xf(centerline[:,2])
yf = sym.lambdify(x, yexpr, 'numpy')
coorKrig[:,1] = yf(centerline[:,2])

def unitVector(vector, ax = 1):
    """ Returns the unit vector of the vector.  """
    norm = np.zeros(vector.shape)
    normValue = np.linalg.norm(vector, axis = ax)
    for i in np.arange(vector.shape[0]):
        norm[i,:] = normValue[i]
    return vector / norm

tangent = np.zeros(centerline.shape)
xexprDeri = sym.diff(xexpr, x)
xderf = sym.lambdify(x, xexprDeri, 'numpy')
tangent[:,0] = xderf(centerline[:,2]+0.0001)
yexprDeri = sym.diff(yexpr, x)
yderf = sym.lambdify(x, yexprDeri, 'numpy')
tangent[:,1] = yderf(centerline[:,2]+0.0001)
tangent[:,2] = 1  # z direction (iSlice)

tangent = unitVector(tangent, ax = 1)


fig = plt.figure()
ax = fig.add_subplot()   # fig.add_subplot(projection='3d')
ax.scatter(centerline[:,2], centerline[:,0], marker='o', label = 'XC')
ax.scatter(centerline[:,2], centerline[:,1], marker='^', label = 'YC')
ax.plot(centerline[:,2], coorKrig[:,0],  '-', label = 'Centerline')
ax.plot(centerline[:,2], coorKrig[:,1], '--', label = 'Kriging')
ax.set_xlabel('Slice')
ax.set_ylabel('XC & YC/voxel')
##ax.set_xlim(0, 275)
##ax.set_ylim(225, 275)
ax.legend(loc='best')

# Error plot
fig2 = plt.figure()
ax2 = fig2.add_subplot()
dist = np.sqrt((centerline[:,0] - coorKrig[:,0])**2 + (centerline[:,1] - coorKrig[:,1])**2)
ax2.plot(centerline[:,2], dist, '-', label = 'error')

ax2.set_xlabel('Slice')
ax2.set_ylabel('local error/voxel')

iSlice, yerror = ax2.get_lines()[0].get_data()
boo = yerror>0.75

for i in iSlice[boo]:
    indexErr = np.argwhere(iSlice == i)
    print(indexErr[0][0])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

# 数据压缩 (nuggetEffect=0)
# 用参数方程求切线