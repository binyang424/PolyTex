# https://pypi.org/project/itk-cleaver/

import xml.dom.minidom
import numpy as np
from skimage import io
import nrrd

### input
filename = 'testdata.nrrd'
nx, ny, nz = 100, 100, 50  # number of cells in each direction
nTow = 51
paraview = 0
inputPath = 'C:/Users/palme/Desktop/test.vtu'
outputPath = "C:/Users/palme/Desktop/"

# load the unstructed grid (voxel model from TexGen with extension of .vtu)
domtree = xml.dom.minidom.parse(inputPath)
rootNode = domtree.documentElement
UnstructuredGrid = rootNode.getElementsByTagName('UnstructuredGrid')

Piece = UnstructuredGrid[0].getElementsByTagName('Piece')
childlist = Piece[0].childNodes

CellData = Piece[0].getElementsByTagName('CellData')
DataArray = CellData[0].getElementsByTagName('DataArray')

# Save vtu as image sequence
for item in DataArray:
    if item.getAttribute('Name') == "YarnIndex":
        YarnIndex = item.childNodes[0].data
        YarnIndex = YarnIndex.split(" ")

        if paraview == 1:
            aa = []
            for i in YarnIndex:
                if '\n' in i:
                    aa.append(i[:-1])
                else:
                    aa.append(i)
            YarnIndex = [i for i in aa if len(i) > 0]
            YarnIndex = [int(x) for x in YarnIndex]

        img_sequence = np.empty([nz, ny, nx])
        for i in range(nz):
            start = i * nx * ny
            end = (i + 1) * nx * ny
            img = np.array(YarnIndex[start:end], dtype=int).reshape([ny, nx])
            img_sequence[i, :, :] = img

data = img_sequence + 1
# np.save( outputPath + 'data.npy', data)

data = np.int32(data)

header = {'space origin': [0, 0, 0],
          "space directions": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          'space': 'left-anterior-superior'}

# Write to a NRRD file
nrrd.write(filename, data, header)
