def save_nrrd(cell_label, mesh_shape, file_path_name):
    """
    Save a mesh to a nrrd file.
    :param cell_label: the cell label of the mesh, type: 1d numpy.ndarray
    :param mesh_shape: number of elements in x, y, and z direction, type: numpy array(int, int, int)
    :param file_path_name: the path and name of the nrrd file, without extension, type: str
    :return: None
    """
    import nrrd
    import numpy as np

    yarnIndex = cell_label

    nx, ny, nz = mesh_shape

    yarnIndex = yarnIndex.reshape((nz, ny, nx))

    data = yarnIndex + 1
    data = np.int32(data)

    header = {'space origin': [0, 0, 0],
              "space directions": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
              'space': 'left-anterior-superior'}

    # Write to a NRRD file with pynrrd
    nrrd.write(file_path_name + ".nrrd", data, header)
    return None
