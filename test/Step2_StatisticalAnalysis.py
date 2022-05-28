import numpy as np
import matplotlib.pyplot as plt
from polykriging import curve2D, utility, statistics
'''
Statistical analysis of points on a yarn surface
'''
resolution = 0.022

windows = 10
bw = 0.004  # specify a bandwidth
# bw = np.arange(0.002, 0.01, 0.001) # specify a range for bandwidth optimization
extremaNum = 35  # number of extrema (control points) for contour description
nuggets = [1e-2]


''' Data loading '''
path = utility.choose_file(titl =
                                "Directory for .npz file containing GeometryFeatures and CoordinatesSorted")
pcdRaw = np.load(path)

# geomFeature = [Area, Perimeter, Width, Height, AngleRotated, Circularity,
#       centroidX, centroidY, centroidZ]
# coordinateSorted = [distance, normalized distance, angular position (degree),
#       coordinateSorted(X, Y, Z)]
coordinatesSorted = pcdRaw["coordinatesSorted.npy"]
geomFeatures = pcdRaw["geomFeatures.npy"]

##coordinatesSorted[:,-3] += 4
##coordinatesSorted[:,-2] += 2

''' Polar plot: angular position - normalized distance '''
##fig = plt.figure()
##ax = fig.add_subplot(projection='polar')
### ax.set_ylabel('Normalized distance')
### The following angle positions should be in radians.
##ax.scatter(coordinatesSorted[:, 2]/360*2*np.pi, coordinatesSorted[:,1],
##           alpha = 0.7, s = 1 )
### reference line for a circle:
##ax.plot(np.arange(0, 2*np.pi, 2*np.pi/360), np.arange(0,1,1/360), linestyle='--', color = 'red' )      
###plt.show()


'''  Kernel density estimation   '''
slices = np.unique(coordinatesSorted[:,-1]) 
nslices = slices.size
pcd = coordinatesSorted[:, 1]
pcd = np.vstack((pcd, coordinatesSorted[:, -1]/ resolution)).T
kdeOutput, extrema = statistics.movingKDE(pcd, bw, windows, extremaNum)
    
'''   Overfitting test    '''
ii = 0
for iSlice in [250 * 0.022]:
# for iSlice in slices[0:6]:
    print(ii)
    ii += 1
    index = np.where(coordinatesSorted[:,-1] == iSlice)
    indexAvg = int(np.average(index))

    for i in range(extrema[:, 0].size):
        if indexAvg < extrema[i,0]:
            interp = kdeOutput[ kdeOutput[:,0]== i, 1][np.int32(extrema[i, 1:])]
    
    mask = coordinatesSorted[:, -1] == iSlice
    coordinate = coordinatesSorted[:, [1,-3, -2, -1]][mask]
    
    coordinate = curve2D.addPoints(coordinate, threshold = 0.02)
    
    plt.close('all')
    ax1 = plt.subplot(1,1,1)

    for nugget in nuggets:
        # Split the data to improve interpolation quality
        mask1 = coordinate[:, 0] < 0.5
        mask2 = coordinate[:, 0] >= 0.5

##        xinter, xexpr = curve2D.curve1Dinter(coordinate[:, [0, 1] ],
##                                      'lin', "lin", nugget, interp )
##        yinter, yexpr = curve2D.curve1Dinter(coordinate[:, [0, 2] ],
##                                      'lin', "lin", nugget, interp )
        xinter, xexpr  = curve2D.curve1Dinter(coordinate[:, [0, 1] ][mask1],
                                  'lin', "lin", nugget, interp[interp<0.5] )
        yinter, yexpr = curve2D.curve1Dinter(coordinate[:, [0, 2] ][mask1],
                                  'lin', "lin", nugget, interp[interp<0.5]  )
        xinterSplit, _ = curve2D.curve1Dinter(coordinate[:, [0, 1] ][mask2],
                                  'lin', "lin", nugget, interp[interp>=0.5] )
        yinterSplit, _ = curve2D.curve1Dinter(coordinate[:, [0, 2] ][mask2],
                                  'lin', "lin", nugget, interp[interp>=0.5]  )
        xinter = np.hstack((xinter, xinterSplit))
        yinter = np.hstack((yinter, yinterSplit))
        
        #ax1.plot(xinter, yinter, '--', label = str(nugget), linewidth = 1)
        ax1.scatter(xinter, yinter, s = 40, marker = '+', color = 'red')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.invert_yaxis()  #y轴反向
        
    ax1.fill(coordinate[:, 1], coordinate[:, 2], alpha = 0.3, color = 'pink')
    
    plt.legend()    
    ax1.axis("equal")
    plt.show()
