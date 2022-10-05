def save_nrrd(cell_label, file_name, file_path='./'):
    """
    Save a mesh to a nrrd file.
    :param cell_label: the cell label of the mesh, type: numpy array(int, int, int)
    :param file_name: the name of the .nrrd file, type: str
    :param file_path: the save path of the .nrrd file, type: str
    :return: None
    """
    import nrrd
    import numpy as np

    indicator = np.zeros_like(cell_label)
    for i, label in enumerate(np.unique(cell_label)):
        mask = cell_label == label
        indicator[mask] = i

    header = {'space origin': [0, 0, 0],
              "space directions": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
              'space': 'left-anterior-superior'}

    # Write to a NRRD file with pynrrd
    if file_name[-5:] != '.nrrd':
        file_name += '.nrrd'
    nrrd.write(file_path + file_name, indicator, header, index_order='C')
    return indicator
