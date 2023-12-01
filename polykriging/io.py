# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import os
import pickle
import re
import logging
import shutil
import copy
import glob

from tkinter import Tk, filedialog, messagebox
from tqdm import trange
from itertools import cycle

import numpy as np
import pandas as pd
import pyvista as pv
import sympy
import vtk
from numpy.compat import os_fspath
from numpy.lib import format
from PIL import Image
from scipy.interpolate import RectBivariateSpline

from .kriging.curve2D import addPoints
from .thirdparty.bcolors import bcolors

file_header = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  8
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/"""

top_separator = """// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

bottom_separator = """
// ************************************************************************* //"""


def write_FoamFile(ver, fmt, cls, location, obj, top_separator):
    return """
FoamFile
{
    version     %.1f;
    format      %s;
    class       %s;
    location    %s;
    object      %s;
}
%s
""" % (ver, fmt, cls, location, obj, top_separator)  # 格式化输出


## boundaryFields for vol<Type>Field
"""
Left: x-
TOP : z +
Front: y +
"""
bFields = """
boundaryField
{
    top
    {
        type            zeroGradient;
    }

    bottom
    {
        type            zeroGradient;
    }

    left
    {
        type            zeroGradient;
    }

    right
    {
        type            zeroGradient;
    }

    back
    {
        type            zeroGradient;
    }

    front
    {
        type            zeroGradient;
    }
}
"""


# Output directory
def mkdir(path):
    """
    Create the output directory if it does not exist.

    Parameters
    ----------
    path: str
        The path of the output directory.

    Returns
    -------
    output_dir: str
        The path of the output directory.
    """
    output_dir = path + "/constant/polyMesh"
    cellDataDir = path + "/0"
    try:
        # os.mkdir(output_dir)  # 创建单层目录
        os.makedirs(output_dir)  # 创建多层目录
        os.makedirs(cellDataDir)
    except FileExistsError:
        logging.warning("Directory already exists. Aborting to be safe.")
    return output_dir


def write_points(points, output_dir='./constant/polyMesh/', scale=1.0):
    """
    Write points to OpenFOAM format

    Parameters
    ----------
    points : array-like
        The points to be written. The shape of the array should be (n, 3).
    output_dir : str, optional
        The directory to store the converted data. The default is './',
        which means the current directory.
    scale : float, optional
        The scale factor to convert the unit of points. The default is 1.0.

    Returns
    -------
     : int
        1 if the writing is successful.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    n_points = points.shape[0]  # np: number of points

    print("Points data writing...")
    points_file = os.path.join(output_dir, "points")

    with open(points_file, "w") as f:
        f.write(file_header)
        f.write(write_FoamFile(2.0, "ascii", "vectorField",
                               "\"constant/polyMesh\"", "points", top_separator))
        f.write("%d\n(\n" % n_points)

        for i in trange(n_points):
            point = points[i] * scale
            f.write("(%f %f %f)\n" % tuple(point))
        f.write(")\n")
        f.write(bottom_separator)
        f.close()
    print("    Points data writing finished.")

    return 1


def write_cell_data(cellDataDict, outputDir='./0/', array_list=None):
    """
    Write cell data to OpenFOAM format

    Parameters
    ----------
    cellDataDict : dict
        A dictionary to store all cell data sets in vtu file using the data set name as key.
    outputDir : str, optional
        The directory to store the converted data. The default is './0/'.
    array_list : list, optional
        A list containing the names of the cell data to be written. The default is None,
    """
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    else:
        shutil.rmtree(outputDir)
        os.makedirs(outputDir)

    keys = cellDataDict.keys()
    print("Cell data writing...")
    for key in keys:
        # skip the cell data not in the array_list if array_list is not None
        if array_list is not None and key not in array_list:
            continue

        print("    - " + key)

        cellData = cellDataDict[key]
        cellDataFile = os.path.join(outputDir, key)
        fileData = open(cellDataFile, "w")
        fileData.write(file_header)

        totalItem = cellData.shape[0]

        if cellData.ndim == 1:
            n_components = 1
            fileData.write(write_FoamFile(2.0, "ascii", "volScalarField", "\"0\"",
                                          key, top_separator))
            fileData.write("dimensions [0 0 0 0 0 0 0];\n")
            fileData.write("internalField   nonuniform List<scalar>\n ")
            fileData.write("%d\n(\n" % totalItem)

        elif cellData.ndim == 2 and cellData.shape[1] == 3:
            n_components = 3
            fileData.write(write_FoamFile(2.0, "ascii", "volVectorField", "\"0\"",
                                          key, top_separator))
            fileData.write("dimensions [0 0 0 0 0 0 0];\n")
            fileData.write("internalField   nonuniform List<vector>\n ")
            fileData.write("%d\n(\n" % totalItem)

        elif cellData.ndim == 2 and cellData.shape[1] == 9:
            n_components = 9
            fileData.write(write_FoamFile(2.0, "ascii", "volTensorField", "\"0\"",
                                          key, top_separator))
            if key == "permeability" or key == "K":
                fileData.write("dimensions [0 2 0 0 0 0 0];\n")
            elif key == "D":
                fileData.write("dimensions [0 -2 0 0 0 0 0];\n")
            fileData.write("internalField   nonuniform List<tensor>\n ")
            fileData.write("%d\n(\n" % totalItem)

        for j in range(totalItem):

            if n_components > 1:
                data = cellData[j]
                fileData.write(str(tuple(data)).replace(",", ""))
            else:
                fileData.write(str(cellData[j]))

            fileData.write("\n")

        fileData.write(");\n")
        fileData.write(bFields)
        fileData.write(bottom_separator)
        fileData.close()


def cell_faces(mesh, ind, neighbor=False):
    """
    Get all faces of a 3D cell.

    Parameters
    ----------
    mesh : vtkUnstructuredGrid
        The volume mesh.
    ind : int
        The cell id.
    neighbor : bool, optional
        If True, return the neighbor cell ids. The default is False.

    Returns
    -------
    faces : list
        A list containing all faces of the cell.
    neighbors : list
        A list containing all neighbor cell ids of the cell.
        Not returned if neighbor is False.
    """
    cell = mesh.GetCell(ind)
    faces = []
    neighbors = set()

    for i in range(cell.GetNumberOfFaces()):
        face_ = cell.GetFace(i)
        point_ids = face_.GetPointIds()
        face = [point_ids.GetId(j) for j in range(face_.GetNumberOfPoints())]
        faces.append(face)

        if neighbor:
            cell_ids = vtk.vtkIdList()
            mesh.GetCellNeighbors(ind, point_ids, cell_ids)
            neighbors.update([cell_ids.GetId(i) for i in range(cell_ids.GetNumberOfIds())])
    if neighbor:
        return faces, list(neighbors)
    else:
        return faces


def get_internel_faces(volume):
    """
    Extract internel faces from a vtkUnstructuredGrid object.

    Parameters
    ----------
    volume : pyvista.UnstructuredGrid
        The volume mesh.

    Returns
    -------
    internal_faces : list
        A list containing all internal faces.
    owner : list
        A list containing all owner cell ids corresponding to the internal faces.
    neighbour : list
        A list containing all neighbour cell ids corresponding to the internal faces.
    """
    print("Internal faces data extracting...")

    n_cells = volume.n_cells
    cells = volume.cells.reshape(n_cells, -1)[:, 1:]

    internal_faces = []
    owner = []
    neighbour = []

    for cell_id in trange(n_cells):
        # print(cell_id)
        cell = cells[cell_id]  # the point ids of the cells

        cell_face_lst, neighbour_cell_ids = cell_faces(volume, cell_id, neighbor=True)
        cell_face_lst = np.array(cell_face_lst)
        cell_face_lst_copy = copy.deepcopy(cell_face_lst)
        cell_face_lst.sort(axis=1)
        neighbour_cell_ids = np.array(neighbour_cell_ids)
        mask = neighbour_cell_ids > cell_id

        for idx in neighbour_cell_ids[mask]:
            face = np.intersect1d(cell, cells[idx, :])

            # sort the row of cell_faces
            face.sort(axis=0)
            mask = np.all(cell_face_lst == face, axis=1)
            face = cell_face_lst_copy[mask][0]

            internal_faces.append(face.tolist())
            owner.append(cell_id)
            neighbour.append(idx)

    print("    Internal faces data extracting finished.")
    return internal_faces, owner, neighbour


def get_boundary_faces(volume):
    """
    Extract boundary faces from a vtkUnstructuredGrid object.

    Parameters
    ----------
    volume : pyvista.UnstructuredGrid
        The volume mesh.

    Returns
    -------
    boundary_faces : numpy.ndarray
        A numpy array containing all boundary faces.
    surf : pyvista.PolyData
        A pyvista surface mesh object.
    """
    print("Boundary faces data extracting...")
    surf = volume.extract_surface(pass_pointid=True, pass_cellid=True)
    surf = surf.cast_to_unstructured_grid()
    vtkOriginalPointIds = surf['vtkOriginalPointIds']
    vtkOriginalCellIds = surf['vtkOriginalCellIds']

    points = surf.points
    cells = surf.cells.reshape(-1, 5)[:, 1:]

    ncells = surf.n_cells
    cellCenters = surf.cell_centers().points  # the centers of each cell

    # Normal of the three coordinate planes
    map = {"YZ": (1, 0, 0), "XZ": (0, 1, 0), "XY": (0, 0, 1)}
    axis = [map.get("YZ"), map.get("XZ"), map.get("XY"), ]

    bbox = surf.bounds  # the bounding box of the surface mesh
    centroid = surf.center  # the center of the surface mesh

    face_boundary = np.zeros((ncells, 4), dtype=int)
    owner_boundary = np.zeros(ncells, dtype=int)

    for i in trange(ncells):
        originalCellId = vtkOriginalCellIds[i]
        originalPointId = vtkOriginalPointIds[cells[i]]

        cell_face_lst = np.array(cell_faces(volume, originalCellId))
        cell_face_lst_copy = copy.deepcopy(cell_face_lst)

        # sort the row of cell_faces
        cell_face_lst.sort(axis=1)
        originalPointId.sort(axis=0)

        mask = np.all(cell_face_lst == originalPointId, axis=1)

        face_boundary[i] = cell_face_lst_copy[mask]

        owner_boundary[i] = originalCellId

    vector_in_face = points[cells[:, 0]] - cellCenters

    mask_center_x = cellCenters[:, 0] < centroid[0]
    mask_center_y = cellCenters[:, 1] < centroid[1]
    mask_center_z = cellCenters[:, 2] < centroid[2]

    mask_yz = np.dot(vector_in_face, axis[0]) == 0
    mask_xz = np.abs(np.dot(vector_in_face, axis[1])) == 0
    mask_xy = np.dot(vector_in_face, axis[2]) == 0

    face_boundary_left = face_boundary[mask_yz & mask_center_x]
    face_boundary_right = face_boundary[mask_yz & ~mask_center_x]

    face_boundary_back = face_boundary[mask_xz & mask_center_y]
    face_boundary_front = face_boundary[mask_xz & ~mask_center_y]
    face_boundary_bottom = face_boundary[mask_xy & mask_center_z]
    face_boundary_top = face_boundary[mask_xy & ~mask_center_z]

    owner_boundary_left = owner_boundary[mask_yz & mask_center_x]
    owner_boundary_right = owner_boundary[mask_yz & ~mask_center_x]
    owner_boundary_back = owner_boundary[mask_xz & mask_center_y]
    owner_boundary_front = owner_boundary[mask_xz & ~mask_center_y]
    owner_boundary_bottom = owner_boundary[mask_xy & mask_center_z]
    owner_boundary_top = owner_boundary[mask_xy & ~mask_center_z]

    face_boundary_dict = {"left": face_boundary_left, "right": face_boundary_right,
                          "back": face_boundary_back, "front": face_boundary_front,
                          "bottom": face_boundary_bottom, "top": face_boundary_top}

    owner_boundary_dict = {"left": owner_boundary_left, "right": owner_boundary_right,
                           "back": owner_boundary_back, "front": owner_boundary_front,
                           "bottom": owner_boundary_bottom, "top": owner_boundary_top}

    return face_boundary_dict, owner_boundary_dict


def write_cell_zone(cell_zone, output_dir='./constant/polyMesh/', ):
    """
    Write the cells to a file for porous properties setting.

    Parameters
    ----------
    cell_zone : dict
        A dict containing all cell ids corresponding to the cell zones. The key
        is the cell zone name and the value (a list) is the cell ids in the zone.
    output_dir : str, optional
        The output directory. The default is './constant/polyMesh/'.

    Returns
    -------
    None.
    """
    if not isinstance(cell_zone, dict):
        print("The type of cell_zone should be dict with key as "
              "the zone name and value (list) as cell ids in the zone.")
        print("Owner of faces data writing failed.")
        return 0

    num_zones = len(cell_zone)

    print("Cell zone data writing...")
    zone_file = open(os.path.join(output_dir, "cellZones"), "w")
    zone_file.write(file_header)
    zone_file.write(write_FoamFile(2.0, "ascii", "regIOobject", "\"constant/polyMesh\"", "cellZones", top_separator))
    zone_file.write("%d\n(\n" % num_zones)

    """ Write cell zones as dictionary """
    for key, value in cell_zone.items():
        zone_file.write("%s\n{\n    type cellZone; \ncellLabels      List<label>\n" % key)
        zone_file.write(str(len(value)) + "\n( \n")
        for cell_id in value:
            zone_file.write(str(cell_id) + '\n')
        zone_file.write(")\n;\n}\n")

    zone_file.write(")\n")
    zone_file.write(bottom_separator)
    zone_file.close()
    print("    Cell zone data writing finished.")
    return num_zones


def write_neighbors(neighbour, output_dir='./constant/polyMesh/'):
    """
    Write the neighbors file.

    Parameters
    ----------
    neighbour : list
        A list containing all neighbor cell ids corresponding to the internal faces.
    output_dir : str, optional
        The output directory. The default is './constant/polyMesh/'.

    Returns
    -------
    None.
    """
    print("Neighbors data writing...")
    neighbour_file = open(os.path.join(output_dir, "neighbour"), "w")
    neighbour_file.write(file_header)
    neighbour_file.write(
        write_FoamFile(2.0, "ascii", "labelList", "\"constant/polyMesh\"", "neighbour", top_separator))
    neighbour_file.write("%d\n(\n" % len(neighbour))
    for i in neighbour:
        neighbour_file.write(str(i) + '\n')
    neighbour_file.write(")\n")
    neighbour_file.write(bottom_separator)
    neighbour_file.close()
    print("    Neighbors data writing finished.")


def write_owner(owner_internal, owner_boundary, output_dir='./constant/polyMesh/', ):
    """
    Write the owner file.

    Parameters
    ----------
    owner_internal : array-like
        A list containing all owner cell ids corresponding to the internal faces.
    owner_boundary : dict
        A list containing all owner cell ids corresponding to the boundary faces.
    output_dir : str, optional
        The output directory. The default is './constant/polyMesh/'.

    Returns
    -------
    None.
    """
    if not isinstance(owner_internal, np.ndarray):
        owner_internal = np.array(owner_internal)
    if not isinstance(owner_boundary, dict):
        print("The type of owner_boundary should be dict with key as "
              "boundary patch name and value as owner cell id.")
        print("Owner of faces data writing failed.")
        return 0

    total_faces = owner_internal.shape[0]
    for key, value in owner_boundary.items():
        total_faces += len(value)

    print("Owner of faces data writing...")
    owner_file = open(os.path.join(output_dir, "owner"), "w")
    owner_file.write(file_header)
    owner_file.write(write_FoamFile(2.0, "ascii", "labelList", "\"constant/polyMesh\"", "owner", top_separator))
    owner_file.write("%d\n(\n" % total_faces)
    """ Write internal face owners first """
    for i in range(len(owner_internal)):
        owner_file.write(str(owner_internal[i]) + '\n')

    """ Write boundary face owners """
    for key, value in owner_boundary.items():
        for cell_id in value:
            owner_file.write(str(cell_id) + '\n')

    owner_file.write(")\n")
    owner_file.write(bottom_separator)
    owner_file.close()
    print("    Owner of faces data writing finished.")
    return total_faces


def write_face(face_points):
    """
    Parameters
    ----------
    face_points : list
        A list containing all face node indices.
    """
    return "%d(%s)\n" % (len(face_points), " ".join([str(p) for p in face_points]))


def write_faces(internal_faces, face_boundary, output_dir='./constant/polyMesh/'):
    """
    Write the faces file.

    Parameters
    ----------
    internal_faces : list
        A list containing all internal face node indices.
    face_boundary : dict
        A dict containing all boundary faces. The key is the boundary patch name,
        and the value is a numpy array containing all boundary face node indices.
    output_dir : str, optional
        The output directory. The default is './constant/polyMesh/'.

    Returns
    -------
    None.
    """
    print("faces data writing...")
    total_faces = len(internal_faces)
    for key, value in face_boundary.items():
        total_faces += len(value)

    print("    Total faces: %d" % total_faces)

    faces_file = open(os.path.join(output_dir, "faces"), "w")
    faces_file.write(file_header)
    faces_file.write(write_FoamFile(2.0, "ascii", "faceList", "\"constant/polyMesh\"", "faces", top_separator))
    faces_file.write("%d\n(\n" % total_faces)

    """ Write internal faces first """
    for face in internal_faces:
        faces_file.write(write_face(face[::-1]))

    """ Write boundary faces """
    for key, value in face_boundary.items():
        for face in value:
            faces_file.write(write_face(face[::-1]))

    faces_file.write(")\n")
    faces_file.write(bottom_separator)
    faces_file.close()
    print("    faces data writing finished.")

    return 1


def write_boundary(face_boundary_dict, start_face, output_dir='./constant/polyMesh/', type=None):
    """
    Boundary file writing.

    Parameters
    ----------
    face_boundary_dict : dict
        A dict contains boundary category. The key is the boundary name and the value
        is a numpy array containing all boundary face node indices. The boundary name
        should be the same as the boundary patch name in the boundary file.
    start_face : int
        The start face index of the boundary faces. equal to the number of internal faces.
    output_dir : str, optional
        The output directory. The default is './constant/polyMesh/'.
    type : dict, optional
        The type of each boundary. The default is None. If None, the type of the boundary
        is set as "patch". The key is the boundary name and the value is the boundary type.
        The key should be the same as the face_boundary_dict.

        The boundary type can be "patch", "wall", "empty", "symmetryPlane", "wedge", "cyclic",
        etc. See OpenFOAM user guide for more details.

    Returns
    -------
    None.
    """
    print("Boundary data writing...")
    n_boundaries = len(face_boundary_dict)

    if type is None:
        type = {}
        for key in face_boundary_dict.keys():
            type[key] = "patch"
    elif not isinstance(type, dict):
        print("The type of type should be dict with key as "
              "boundary patch name and value as boundary type.")
        print("Boundary data writing failed.")
        return 0

    boundary_file = open(os.path.join(output_dir, "boundary"), "w")
    boundary_file.write(file_header)
    boundary_file.write(
        write_FoamFile(2.0, "ascii", "polyBoundaryMesh", "\"constant/polyMesh\"", "boundary", top_separator))
    boundary_file.write("%d\n(\n    " % n_boundaries)

    for key in list(face_boundary_dict.keys()):
        print("    - " + key)
        boundary_file.write("""%s
    {
        type %s;
        nFaces %d;
        startFace %d;
    }
    """ % (key, type[key], len(face_boundary_dict[key]), start_face))
        start_face += len(face_boundary_dict[key])

    boundary_file.write(")\n")
    boundary_file.write(bottom_separator)
    boundary_file.close()
    print("    Boundary data writing finished.")

    return 1


def voxel2foam(mesh, scale=1, outputDir="./", boundary_type=None, cell_data_list=None) -> None:
    """
    Convert a voxel mesh to OpenFOAM mesh. The cell data is converted to OpenFOAM initial conditions
    and saved in the 0 timestep folder.

    Parameters
    ----------
    mesh : pyvista.UnstructuredGrid
        The voxel mesh.
    scale : float, optional
        The scale factor to convert the unit of points. The default is 1.0.
    outputDir : str, optional
        The output directory. The default is './'.
    boundary_type : dict, optional
        The type of each boundary. The default is None. If None, the type of the boundary
        is set as "patch". The key is the boundary name and the value is the boundary type.
        The key should be the same as the face_boundary_dict.

        The boundary type can be "patch", "wall", "empty", "symmetryPlane", "wedge", "cyclic",
        etc. See OpenFOAM user guide for more details.

    Returns
    -------
    None.
    """
    cwd = os.getcwd()
    print("Current working directory: ", cwd)

    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError("mesh must be a pyvista.UnstructuredGrid. If you have a vtu file, "
                        "use pyvista.read('filename.vtu') to read the file.")

    # Create the output directory if it does not exist.
    if not os.path.exists(outputDir):
        path = mkdir(outputDir)
    else:
        path = outputDir

    os.chdir(outputDir)  # set the output directory as the current working directory

    path_polyMesh = "./constant/polyMesh"
    path_0 = "./0"

    """ 1. Write points """
    pts = mesh.points  # numpy.ndarray (npts, 3)
    write_points(pts, output_dir=path_polyMesh, scale=scale)

    """ 2. Write cell data """
    cellDataDict = mesh.cell_data
    write_cell_data(cellDataDict, outputDir=path_0, array_list=cell_data_list)

    """ 3. Write faces """
    # Get internal faces
    internal_faces, owner_internal, neighbour = get_internel_faces(mesh)

    # Get boundary faces
    face_boundary_dict, owner_boundary_dict = get_boundary_faces(mesh)

    """ 4. Write cell zones for porous region """
    cell_zone = {"porousLayer": np.arange(mesh.n_cells)[mesh.cell_data["porosity"] < 0.995]}
    write_cell_zone(cell_zone, output_dir=path_polyMesh)

    """ 5. Write neighbor file """
    write_neighbors(neighbour, output_dir=path_polyMesh)

    """ 6. Write owner file """
    write_owner(owner_internal, owner_boundary_dict, output_dir=path_polyMesh)

    """ 7. Write face file """
    write_faces(internal_faces, face_boundary_dict, output_dir=path_polyMesh)

    """ 8. Write boundary file """
    write_boundary(face_boundary_dict, len(internal_faces), output_dir=path_polyMesh, type=boundary_type)

    """ 9. Write boundary conditions """

    print("Mesh writing finished!")
    os.chdir(cwd)  # set the current working directory back to the original directory
    return None


def case_preparation(src, dst, verbose=False):
    """
    Copy files from the template case to the output directory

    Parameters
    ----------
    src : str
        The path of the template case.
    dst : str
        The path of the output case.
    """

    if not os.path.exists(dst + "system"):
        os.makedirs(os.path.join(dst, "system"))

    # system directory
    files = glob.glob(src + "system/*")
    for file in files:
        if os.path.isdir(file):
            continue
        if not os.path.exists(os.path.join(dst + file.split("/")[-1])):
            if verbose:
                print("Copying {} ...".format(file.split("/")[-1]))
            shutil.copy(file, dst + "system/")
        else:
            print("File {} already exists.".format(file.split("/")[-1]))

    # constant directory
    files = glob.glob(src + "constant/*")
    for file in files:
        if os.path.isdir(file):  # neglect directories such as polyMesh
            continue
        # if not exist in the destination directory, copy the file
        if not os.path.exists(os.path.join(dst, file.split("/")[-1])):
            if verbose:
                print("Copying {} ...".format(file.split("/")[-1]))
            shutil.copy(file, dst + "constant/")
        else:
            print("File {} already exists.".format(file.split("/")[-1]))

    # 0 directory
    files = glob.glob(src + "0/*")
    for file in files:
        print(os.path.join(dst, file.split("/")[-1]))
        if not os.path.exists(os.path.join(dst, file.split("/")[-1])):
            if verbose:
                print("Copying {} ...".format(file.split("/")[-1]))
            shutil.copy(file, dst + "0/")
        else:
            print("File {} already exists.".format(file.split("/")[-1]))

    # root directory
    files = glob.glob(src + "*")
    # copy the files in the root directory to the destination directory and neglect directories
    for file in files:
        if os.path.isdir(file):
            continue
        if not os.path.exists(os.path.join(dst, file.split("/")[-1])):
            # this check here seems does not work, why?
            if verbose:
                print("Copying {} ...".format(file.split("/")[-1]))
            shutil.copy(file, dst)
        else:
            print("File {} already exists.".format(file.split("/")[-1]))


def voxel2img(mesh, mesh_shape, dataset="YarnIndex", save_path="./img/",
              scale=None, img_name="img", format="tif", scale_algrithm="linear"):
    """
    Convert a voxel mesh to a series of images.
    Parameters
    ----------
    mesh : pyvista.UnstructuredGrid
        The voxel mesh to convert.
    mesh_shape : list
            The number of cells in each direction of the mesh [nx, ny, nz].
    dataset : str, optional
            The name of the cell data to convert. The default is "YarnIndex".
    save_path : str, optional
            The path to save the images. The default is "./img/".
    scale : int
            The scale factor of the image. The default is None.
    img_name : str, optional
            The name of the output image. The default is "img". The slice
            number will be added to the end of the name and separated by
            an underscore.
    format : str, optional
            The format of the output image. The default is "tif".
    scale_algrithm : str, optional
            The algorithm used to scale the pixel numbers of the image.
            The default is "linear". The other option is "spline".

            TODO: The "spline" algorithm is only working for x and y directions yet.
                  The z direction is to be implemented.

    Returns
    -------
    None

    Examples
    --------
    >>> import pyvista as pv
    >>> import polykriging as pk
    >>> mesh = pv.read("./v2i.vtu")
    >>> mesh_shape = [20, 20, 5]
    >>> pk.io.voxel2img(mesh, mesh_shape, dataset="YarnIndex",
                        save_path="./img/",
                        scale=50, img_name="img", format="tif",
                        scale_algrithm="linear")
    """
    nx, ny, nz = mesh_shape  # number of cells in each direction

    yarnIndex = mesh.cell_data[dataset] + 1
    yarnIndex = yarnIndex / np.max(yarnIndex) * 255

    img_sequence = np.reshape(yarnIndex, [nz, nx, ny])

    # print("The shape of the mesh is: ", mesh_shape)

    x = np.arange(0, ny)
    y = np.arange(0, nx)

    if scale is not None:
        nx2 = nx * scale
        ny2 = ny * scale
        x2 = np.linspace(0, ny, ny2, endpoint=True)
        y2 = np.linspace(0, nx, nx2, endpoint=True)

        if scale_algrithm == "linear":
            for i in range(3):
                img_sequence = np.repeat(img_sequence, scale, axis=i)

            nx = nx * scale
            ny = ny * scale
            nz = nz * scale

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # check if the save path is empty. If not, print a warning
    if os.listdir(save_path):
        print(os.listdir(save_path))
        print("Warning: The save path is not empty! The images with the "
              "same name will be overwritten!")

    for i in range(nz):
        img = img_sequence[i, :, :]

        # interpolate the numpy array to get a smooth image
        if scale is not None and scale_algrithm == "spline":
            img = RectBivariateSpline(x, y, img, kx=3, ky=3)(x2, y2)

            # save the image
        img = Image.fromarray(img)
        img = img.convert('L')

        path = os.path.join(save_path, img_name + "_" + str(i) + "." + format)
        img.save(path)

    return None


def construct_tetra_vtk(points, cells, save=None, binary=True):
    """
    Construct a UnstructuredGrid tetrahedral mesh from vertices and connectivity.

    Parameters
    ----------
    points: (n, 3) array
        vertices
    cells: (m, 4) array
        connectivity
    save: str
        The path and file name of the vtk file to be saved ("./tetra.vtk").
        If None, the vtk file will not be saved.
    binary: bool
        whether to save the mesh in binary format

    Returns
    -------
    grid: pyvista.UnstructuredGrid
        UnstructuredGrid tetrahedral mesh
    """
    n_cells = cells.shape[0]
    offset = np.array([4 * i for i in np.arange(n_cells)])
    cells = np.concatenate(np.insert(cells, 0, 4, axis=1)).astype(np.int64)
    cell_type = np.array([vtk.VTK_TETRA] * n_cells)
    grid = pv.UnstructuredGrid(offset, cells, cell_type, np.array(points))
    if save is not None:
        grid.save(save, binary=binary)
    return grid


def cwd_chdir(path=""):
    """
    Set given directory or the folder where the code file is as current working directory

    Parameters
    ----------
    path:
        the path of current working directory. if empty, the path of the code file is used.

    Returns
    -------
    cwd: the current working directory.
    """
    import sys

    if path == "":
        cwd = str(sys.argv[0])
        cwd, pyname = os.path.split(cwd)
    else:
        cwd = path
    os.chdir(cwd)
    return cwd


def choose_directory(titl='Select the target directory:'):
    """
    Choose a directory with GUI and return its path.

    Parameters
    ----------
    titl: String.
        The title of the open folder dialog window.

    Returns
    -------
    path: String.
        The path of the selected directory.
    """

    print(titl)
    # pointing root to Tk() to use it as Tk() in program.
    # like a window (container) where we can put widgets.
    directory_root = Tk()
    directory_root.withdraw()  # Hides small tkinter window.
    directory_root.attributes('-topmost',
                              True)  # Opened windows will be active. above all windows.
    path_work = filedialog.askdirectory(title=titl)  # Returns opened path as str
    if path_work == '':
        top = Tk()
        top.withdraw()
        top.geometry("150x150")
        message = messagebox.askquestion("Warning", "You did not select any folder! "
                                                    "Do you wish to select again?")
        if message == 'yes':
            return choose_directory()
        elif message == 'no':
            return None
    else:
        # replace the forward slash returned by askdirectory
        # with backslash (\) on Windows.
        return path_work.replace('/', os.sep)


def filenames(path, filter="csv"):
    """
    Get the list of files in the given folder.

    Parameters
    ----------
    path:
        the path of the folder
    filter:
        filter for file selection.++

    Returns
    -------
    flst: the list of files in the given folder.
    """
    filenamels = os.listdir(path)
    # filter the file list by the given filter.
    flst = [x for x in filenamels if (filter in x)]
    flst.sort()
    return flst


def zip_files(directory, file_list, filename, remove="True"):
    """
    Add multiple files to the zip file.

    Parameters
    ----------
    directory: String.
        The directory of the files to be added to zip file. Therefore,
        all the files in the file_list should be in the same directory.
    file_list : List.
        The list of file names to be added to the zip file (without directory).
    filename: String.
        The name of the zip file. The zip file is saved in the same directory
    remove:
        Whether to remove original files after adding to zip file.
        Default is True. If False, the original files will not be removed.

    Returns
    -------
    None.
    """
    from zipfile import ZipFile

    # check extension of the zip file
    if filename[-4:] != ".zip":
        filename += ".zip"

    with ZipFile(filename, 'w') as zipObj:
        for i in range(len(file_list)):
            zipObj.write(directory + file_list[i])

            if remove == "True":
                os.remove(directory + file_list[i])
    print(bcolors.ok("Zip file saved as " + filename))


def choose_file(titl='Select the target directory:', format='csv'):
    """
    Choose a file with GUI and return its path.

    Parameters
    ----------
    titl: String.
        The title of the window.

    Returns
    -------
    path: String.
        The path of the file.
    """

    print(titl)
    directory_root = Tk()
    directory_root.withdraw()  # Hides small tkinter window.
    directory_root.attributes('-topmost',
                              True)  # Opened windows will be active (appears above all windows)
    path_work = filedialog.askopenfilename(
        title=titl, filetypes=[(format, format), ('All files', '*.*')])  # Returns opened path as str

    # replace the forward slash returned by askdirectory
    # with backslash (\) on Windows.
    return path_work.replace('/', os.sep)


def save_nrrd(cell_label, file_name, file_path='./'):
    """
    Save the labels of a hexahedral mesh to a nrrd file. The labels should be
    starting from 0 and increasing by 1.

    Parameters
    ----------
    cell_label: numpy array(int, int, int)
        The cell label of the mesh.
    file_name: String
        The name of the .nrrd file.
    file_path: String
        The save path of the .nrrd file.

    Returns
    -------
    None
    """
    import nrrd

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


class save_krig(dict):
    """
    This class saves a dictonary of sympy expressions to a file in human
    readable form and then load as sympy expressions directly without other
    conversion. It is called by polykriging.fileio.pk_save to save kriging
    expressions to a ".krig" file and by polykriging.fileio.pk_load to load
    these files. Therefore, the class is not intended to be used directly by
    the user.

    Note:
    --------
    This class is taken from: https://github.com/sympy/sympy/issues/7974.
    A bug in exec() is fixed and some modifications are made to make it
    fit for the purpose of this project (store the kriging expression).

    Example:
    --------
    >>> import sympy
    >>> from polykriging.fileio.save_krig import save_krig
    >>> a, b = sympy.symbols('a, b')
    >>> d = save_krig({'a':a, 'b':b})
    >>> d.save('name.krig')
    >>> del d
    >>> d2 = save_krig.load('name.krig')
    """

    def __init__(self, *args, **kwargs):
        super(save_krig, self).__init__(*args, **kwargs)

    def __repr__(self):
        d = dict(self)
        for key in d.keys():
            d[key] = sympy.srepr(d[key])
        # regex is just used here to insert a new line after
        # each dict key, value pair to make it more readable
        return re.sub('(: \"[^"]*\",)', r'\1\n', d.__repr__())

    def save(self, file):
        with open(file, 'w') as savefile:
            savefile.write(self.__repr__())

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r') as loadfile:
            # Note that the variable name temp should not be the same as the other
            # local variables in the function, otherwise exec will not work and will
            # raise an NameError: name 'temp' is not defined.
            exec("temp =" + loadfile.read())
        # convert the strings back to sympy expressions and return a new save_krig.
        # This is done by calling the save_krig constructor with the new dict.
        # locals() is used to get the sympy symbols from the exec statement above.
        d = locals()['temp']
        for key in d.keys():
            d[key] = sympy.sympify(d[key])
        return cls(d)


def save(file, arr, allow_pickle=True, fix_imports=True):
    """
    This is an exact copy of numpy.save, except that it does not check the extensions.

    Parameters
    ----------
    file : file, str, or pathlib.Path. File or filename to which the data is saved.
    arr : array_like. Array data to be saved.
    allow_pickle : bool, optional
        Allow saving object arrays using Python pickles. Reasons for disallowing
        pickles include security (loading pickled data can execute arbitrary
        code) and portability (pickled objects may not be loadable on different
        Python installations, for example if the stored objects require libraries
        that are not available, and not all pickled data is compatible between
        Python 2 and Python 3).
        Default: True
    fix_imports : bool, optional
        Only useful in forcing objects in object arrays on Python 3 to be
        pickled in a Python 2 compatible way. If `fix_imports` is True, pickle
        will try to map the new Python 3 names to the old module names used in
        Python 2, so that the pickle data stream is readable with Python 2.
    """

    if hasattr(file, 'write'):
        file_ctx = contextlib.nullcontext(file)
    else:
        file = os_fspath(file)
        # if not file.endswith('.npy'):
        #     file = file + '.npy'
        file_ctx = open(file, "wb")

    with file_ctx as fid:
        arr = np.asanyarray(arr)
        format.write_array(fid, arr, allow_pickle=allow_pickle,
                           pickle_kwargs=dict(fix_imports=fix_imports))


def pk_save(fp, data):
    """
    Save a Python dict or pandas dataframe as a file format defined in polykriging (.coo, geo) file

    Parameters
    ----------
    fp: str
        File path and name to which the data is saved. If the file name does not end with
        a supported file extension, a ValueError will be raised.
    data: Tow, Tex, or dict
        The data to be saved. It can be several customised file formats for polykriging.


    Returns
    -------
    None
    """
    filename = os.path.basename(fp)
    # get file extension
    ext = os.path.splitext(filename)[1]

    if ext == "":
        raise ValueError("The file extension is not given. Supported file extensions are "
                         ".coo, .geo, .tow, .tex, and .krig.")
    elif ext not in ['.coo', '.geo', '.tow', '.tex', '.krig']:
        raise ValueError("The file extension is not supported. Supported file extensions are "
                         ".coo, .geo, .tow, .tex, and .krig.")

    if fp.endswith('.krig'):
        expr = save_krig(data)
        expr.save(fp)
        print(bcolors.ok("The Kriging function {} is saved successfully.").format(filename))
    elif ext in ['.tow', '.tex']:
        with open(fp, 'wb') as f:
            pickle.dump(data, f)
        f.close()
        print(bcolors.ok("The file {} is saved successfully.").format(filename))
    elif isinstance(data, pd.DataFrame):  # save as .coo or .geo file
        data.to_pickle(fp)
        print(bcolors.ok("The file {} is saved successfully.").format(filename))
    else:
        raise TypeError("The input data type is not supported.")


def pk_load(file):
    """
    Load a file format defined in polykriging (.coo, .geo, or .stat) file
    and return as a pandas dataframe or a numpy.array object.

    Parameters
    ----------
    file:  str, or pathlib.Path.
        File path and name to which the data is stored.

    Returns
    -------
    df: pandas.DataFrame or numpy.ndarray
        The data to be loaded. It is a pandas dataframe if the file is a .coo/geo file.
        Otherwise, it is a numpy array or dict and a warning will be raised.
    """

    filename = os.path.basename(file)
    ext = os.path.splitext(filename)[1]

    if ext == "":
        raise ValueError("The file extension is not given. Supported file extensions are "
                         ".pcd, .coo, .geo, .tow, .tex, and .krig.")
    elif ext not in ['.pcd', '.coo', '.geo', '.tow', '.tex', '.krig']:
        raise ValueError("The file extension is not supported. Supported file extensions are "
                         ".pcd, .coo, .geo, .tow, .tex, and .krig.")

    if file.endswith('.krig'):
        print(bcolors.ok("The Kriging expression {} is loaded successfully.").format(filename))
        return save_krig.load(file)

    if ext in ['.coo', '.geo']:
        data = pd.read_pickle(file)
        print(bcolors.ok("The file {} is loaded successfully.").format(filename))
    elif ext in ['.tow', '.tex']:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        f.close()
    else:
        data = np.load(file, allow_pickle=True, fix_imports=True).tolist()
        print(bcolors.ok("The file {} is loaded successfully.").format(filename))

    return data


def pcd_to_ply(file_pcd, file_ply, binary=False):
    """
    Convert a pcd file to ply file.

    Parameters
    ----------
    file_pcd : str
        The path of the pcd file or pathlib.Path. File or filename to which the data is saved.
    file_ply : str
        The path of the ply file or pathlib.Path. File or filename to which the data is to be saved.
    :return: None
    """
    import meshio

    vertices = pk_load(file_pcd).to_numpy()

    mesh = meshio.Mesh(points=vertices[:, 1:],
                       cells=[],
                       # Optionally provide extra data on points, cells, etc.
                       point_data={"nx": vertices[:, 0], "ny": vertices[:, 1], "nz": vertices[:, 2]},
                       # Each item in cell data must match the cells array
                       # cell_data={"a": [[0.1, 0.2], [0.4]]},
                       )

    meshio.write(file_ply, mesh, binary=False)


def coo_to_ply(file_coo, file_ply, interpolate=False, threshold=0.1):
    """
    Convert a pcd file to ply file.

    Parameters
    ----------
    file_coo : str
        The path of the coo file or pathlib.Path. File or filename to which the data is saved.
    file_ply : str
        The path of the ply file or pathlib.Path. File or filename to which the data is to be saved.
    :return: None
    """
    import meshio
    df = pk_load(file_coo)
    vertices = df.to_numpy()[:, [1, 3, 4, 5]]

    if interpolate:
        # interpolate
        vertices = addPoints(vertices, threshold=threshold)

    mesh = meshio.Mesh(points=vertices[:, 1:],
                       cells=[],
                       # Optionally provide extra data on points, cells, etc.
                       point_data={"nx": vertices[:, 0], "ny": vertices[:, 1], "nz": vertices[:, 2]},
                       # Each item in cell data must match the cells array
                       # cell_data={"a": [[0.1, 0.2], [0.4]]},
                       )

    meshio.write(file_ply, mesh, binary=False)


def save_ply(file, vertices, cells=[], point_data={}, cell_data={}, binary=False):
    print(bcolors.warning(
        "This function will be deprecated in the future. Please use polykriging.meshio_save() instead."))
    return meshio_save(file, vertices, cells, point_data, cell_data, binary)


def meshio_save(file, vertices, cells=[], point_data={}, cell_data={}, binary=False):
    """
    Save surface mesh as a mesh file by definition of vertices and faces. Point data and cell data can be added.
    It is a wrapper of meshio.write() function.

    Parameters
    ----------
    file : str
        The path of the ply file or pathlib.Path. File or filename to which the data is saved.
    vertices : numpy.ndarray
        The vertices of the mesh. The shape of the array is (n, 3), where n is the number of vertices.
    cells : list, optional
        The faces of the mesh stored as the connectivity between vertices. The default is [].
    point_data : dict, optional
        The point data of the mesh. The default is {}.
    cell_data : dict, optional
        The cell data of the mesh. The default is {}. Note that the cell data should be added as a
        list of arrays. Each array in the list corresponds to a cell type. For example, if the mesh
        has 2 triangles and 1 quad, namely,
          cells = [("triangle", [0, 1, 2], [1,2,3]), ("quad", [3, 4, 5, 6])],
        then the cell data should be added as
          cell_data = {"data": [[1, 2], [3]}.
    binary : bool, optional
        If True, the data is written in binary format. The default is False.

    Returns
    -------
    None.

    Examples
    --------
    >>> import numpy as np
    >>> import polykriging as pk
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> cells = [("triangle", [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])]
    >>> point_data = {"a": np.array([0, 1, 2, 3])}
    >>> cell_data = {"b": np.array([[0, 1, 2, 3],])}
    >>> pk.meshio_save("test.ply", vertices, cells, point_data, cell_data)
    >>> print("Done")
    Done

    """
    try:
        import meshio
    except ModuleNotFoundError:
        raise ImportError("This function requires meshio package but it is not installed.")

    mesh = meshio.Mesh(points=vertices,
                       cells=cells,
                       # Optionally provide extra data on points, cells, etc.
                       point_data=point_data,
                       # Each item in cell data must match the cells array
                       cell_data=cell_data,
                       )
    # check if the extension in file is in [ply, stl, vtk, vtu]
    import os
    filename, file_extension = os.path.splitext(file)
    if file_extension in [".ply", ".stl", ".vtk", ".vtu"]:
        meshio.write(file, mesh, binary=binary)
        print("The mesh is saved as " + filename + file_extension + " file successfully.")
    else:
        raise ValueError("The file extension is not supported. "
                         "Please use ply, stl, vtk, or vtu.")

    return None


def get_ply_property(mesh_path, column, skip=11, type="vertex", save_vtk=False):
    """
    This function get a vertex property or cell property from a mesh stored as .ply
    format. It is intended to be used to get the user-defined properties that most of
    meshing and rendering software does not support.

    Note
    ----
    The mesh must be saved as ASCII format.

    Parameters
    ----------
    mesh_path : str
        The path of the mesh file with .ply extension.
    column : int or list of int
        The column number of the property.
    skip : int, optional
        The number of lines to skip in the header. The default is 11.
    type : str, optional
        The type of the property. The default is "vertex" for vertex property. The other
        possible value is "cell" for cell property.
    save_vtk : bool, optional
        If True, the mesh is saved as a vtk file. The default is False.

    Returns
    -------
    property : numpy.ndarray
        The property of the mesh.

    Examples
    --------
    >>> import polykriging as pk
    >>> mesh_path = "./weft_0_lin_lin_krig_30pts.ply"
    >>> quality = pk.get_ply_property(mesh_path, -2, skip=11, type="vertex", save_vtk=False)
    >>> quality
    """
    import pyvista as pv
    import numpy as np

    mesh = pv.read(mesh_path)
    n_pts = mesh.n_points
    n_cells = mesh.n_cells

    # load as csv
    mesh_txt = np.loadtxt(mesh_path, dtype=object, delimiter=" ", skiprows=11)

    if type == "vertex":
        vertex = mesh_txt[:n_pts]
        quality = vertex[:, column].astype(np.float32)
    elif type == "cell":
        cell = mesh_txt[n_pts:n_pts + n_cells]
        quality = cell[:, column].astype(np.float32)

    if save_vtk:
        mesh["quality"] = quality
        mesh.save(mesh_path[:-4] + ".vtk")

    return quality


def save_csv(filename, dataset, csv_head):
    """
    Save numpy array to csv file with given info in the first row.

    Parameters
    ----------
    filename:
        The path and name of the csv file.
    dataset: List or numpy.ndarray
        The dataset to be saved in the csv file
    csv_head:
        A list of headers of the csv file. The length of the list
        should be the same as the number of columns in the dataset.

    Returns
    -------
    None.

    """
    import csv

    if filename[-4:] != ".csv":
        filename = filename + ".csv"

    path = filename + ".csv"

    with open(path, 'w', newline="") as f:
        csv_write = csv.writer(f)

        csv_write.writerow(csv_head)
        for row in dataset:
            csv_write.writerow(row)
    return 1


if "__main__" == __name__:
    import doctest

    doctest.testmod()
