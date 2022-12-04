"""
To check the use of this module, please refer to the example "mesh_from_image.py"
 in the test folder of Polykriging.

MIT License

    Copyright (c) 2022 Bin Yang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import numpy as np
import pyvista as pv


def im_to_ugrid(im):
    """
    Convert image or image sequence to an unstructured grid.

    Parameters
    ----------
    im : image object
        The image sequence stored as a single tif file.

    Returns
    -------
    ugrid : pyvista.UnstructuredGrid
        The unstructured grid discretized from the image with voxels.
    im_dim : numpy.ndarray
        The image dimension.

    Example
    -------
    >>> import polykriging as pk
    >>> im = pk.example("image")
    >>> mesh, mesh_dim = pk.mesh.im_to_ugrid(im)
    """
    mesh = pv.read(im)
    ugrid = mesh.cast_to_unstructured_grid()
    # image dimension: It works because the default image origin is (0, 0, 0)
    # and the spacing is (1, 1, 1).
    im_dim = np.array(mesh.bounds[1::2], dtype=int) + 1
    return ugrid, im_dim


def mesh_extract(ugrid, threshold, pointdata='Tiff Scalars', type="foreground"):
    """
    Extract part of the mesh from the unstructured grid according to the value of point data.

    Parameters
    ----------
    ugrid : pyvista.UnstructuredGrid
        The unstructured grid discretized from the image with voxels.
    threshold : float
        The threshold value of the point data specified by the parameter 'pointdata'.
    pointdata : str, optional
        The point data name, by default 'Tiff Scalars'.
    type : str, optional
        The type of the extracted mesh, by default "forground". The type can be "forground" or
        "background". If the type is "foreground", the extracted mesh is where the point data is
        greater than the threshold value. If the type is "background", the extracted mesh is where
        the point data is less than the threshold value.

    Returns
    -------
    subset_final : pyvista.UnstructuredGrid
        The extracted volume mesh.
    surf : pyvista.PolyData
        The extracted surface mesh.
    """
    if type == "foreground":
        mask = ugrid[pointdata] > threshold
    elif type == "background":
        mask = ugrid[pointdata] < threshold

    sub_index = np.arange(ugrid.n_points)[mask]
    # Keep the cells that contain any of the points in the sub_index.
    # The subset of the mesh actually contains one layer more cells than the
    # sub_index, so we need to remove the extra cells latter.
    subset = ugrid.extract_points(sub_index)

    # Remove the extra cells according to its surface mesh.
    surf = subset.extract_surface()
    cell_rm = np.unique(surf['vtkOriginalCellIds'])

    # Here we remove the extra cells to avoid having a surface that is "dialated".
    subset_final = subset.remove_cells(cell_rm)
    surf = subset_final.extract_surface()

    return subset_final, surf


def mesh_separation(mesh, plot=False):
    """
    Separate the mesh object into different regions according to the
    connectivity of the mesh. It may not work for mesh with multiple
    regions that are connected.

    Parameters
    ----------
    mesh : pyvista.UnstructuredGrid
        The mesh object.

    Returns
    -------
    mesh_dict : dict
        The dictionary of the separated mesh objects. The key is the
        region number and the value is the mesh object.
    """
    mesh_dict = {}

    conn = mesh.connectivity(largest=False)
    mesh_dict["connectivity"] = conn
    if plot:
        conn.plot(show_edges=True)

    region_ids = conn['RegionId']  # cell data
    region_num = np.unique(region_ids)

    for i in region_num:
        mesh_dict[str(i)] = conn.extract_cells(np.where(region_ids == i)[0])

    return mesh_dict


def get_vcut_plane(surf_mesh, direction='x'):
    """
    Get the vertical cut plane of the surf mesh in the direction
    of x, y, or z axis through (boundary) cutting edge extraction.

    Parameters
    ----------
    surf_mesh : pyvista.PolyData
        The surface mesh.
    direction : str, optional
        The direction of the vertical cut plane, by default 'x'.
        The direction can be 'x', 'y', or 'z'.

    Returns
    -------
    vcut_plane : numpy.ndarray
        The vertical cut planes of the surf mesh.
    trajectory : numpy.ndarray
        The trajectory (centroid) of the vertical cut plane calculated
        by averaging the coordinates of the points on the cutting edge.
    """
    surf_points = surf_mesh.points
    cell_centers = surf_mesh.cell_centers().points

    direct = {"x": 0, "y": 1, "z": 2}
    vcut_planes = np.zeros_like(surf_points)

    if direction not in direct.keys():
        raise ValueError("The direction can only be 'x', 'y', or 'z'.")
    else:
        coo_direct = surf_points[:, direct[direction]]
        slices = np.unique(coo_direct).astype(int)

    num = 0
    for iSlice in slices:
        mask = cell_centers[:, direct[direction]] < iSlice
        try:
            temp = surf_mesh.remove_cells(mask)
            # temp.plot(show_scalar_bar=False, show_edges=False)
            boundary = temp.extract_feature_edges(feature_angle=100,
                                                  boundary_edges=True, non_manifold_edges=False, manifold_edges=False)
            lines = boundary.lines.reshape(-1, 3)[:, 1:]
            pts_idx_sort = __node_sort_curve(lines)[1:]

            boundary_points = boundary.points[pts_idx_sort, :]

            n_boundary_points = boundary.n_points

            vcut_planes[num: num + n_boundary_points, :] = boundary_points
            num += n_boundary_points
        except:
            continue

    # remove the zero rows: we may lose some points at the end of the mesh.
    vcut_planes = vcut_planes[~np.all(vcut_planes == 0, axis=1)]

    trajectory = np.zeros([np.unique(vcut_planes[:, direct[direction]]).size, 3])
    for i, x in enumerate(np.unique(vcut_planes[:, direct[direction]])):
        # get the points that have the same x coordinate
        index = np.where(vcut_planes[:, direct[direction]] == x)[0]
        x_i = vcut_planes[:, 0][index]
        y_i = vcut_planes[:, 1][index]
        z_i = vcut_planes[:, 2][index]
        # find the centroid
        trajectory[i, :] = np.array([x, np.mean(y_i), np.mean(z_i)])

    return np.array(vcut_planes), trajectory


def __node_sort_curve(curve_connectivity):
    """
    Sort the nodes of the curve according to node connectivity.

    Parameters
    ----------
    curve_connectivity : array_like
        The connectivity of the nodes of the curve. The array-like data stores
        the node index of each segment of the curve with two nodes per row.
        Therefore, the shape of the array is (n_segments, 2) like:
        [[node 1,node 2], [node 1, node 2]  .... [node 1, node 2]].

    Returns
    -------
    numpy.ndarray
        The returned has the same shape as the input curve_connectivity. However,
        the nodes are sorted according to their connectivity in the following
        order: [[1, 2], [2, 4], [4, 7] ...].
    """
    # tranverse the lines to get the trajectory according to the connectivity of the lines
    curve_connectivity_ordered = curve_connectivity[0, :]
    lines = np.delete(curve_connectivity, 0, axis=0)

    while lines.shape[0] > 0:
        if curve_connectivity_ordered[0] == curve_connectivity_ordered[-1]:
            break

        n_lines = lines.shape[0]
        n_del = []
        for i in range(n_lines):
            line_i = lines[i, :]
            mask1 = line_i == curve_connectivity_ordered[0]
            mask2 = line_i == curve_connectivity_ordered[-1]
            if any(mask1):
                # insert the other nodes to the beginning of curve_connectivity_ordered
                curve_connectivity_ordered = np.insert(curve_connectivity_ordered, 0, line_i[~mask1])
                # remove the row
                n_del.append(i)
            elif any(mask2):
                # insert the other nodes to the end of curve_connectivity_ordered
                curve_connectivity_ordered = np.append(curve_connectivity_ordered, line_i[~mask2])
                # remove the row
                n_del.append(i)

        lines = np.delete(lines, n_del, axis=0)

    return curve_connectivity_ordered


def slice_plot(vcut_planes, skip=10, marker='o', marker_size=0.1, dpi=300, save=False, save_path=None):
    """
    Plot the vertical cut planes.

    Parameters
    ----------
    vcut_planes : numpy.ndarray
        The vertical cut planes of the surf mesh stored in a numpy array.
        The shape of the array is (n_points, 3).
    skip : int, optional
        The number of cut planes to skip when plotting the vertical cut planes,
        by default 10.
    marker : str, optional
        The marker type, by default 'o'.
    marker_size : float, optional
        The marker size, by default 0.1.
    dpi : int, optional
        The resolution of the figure, by default 300.
    save : bool, optional
        Whether to save the figure, by default False.
    save_path : str, optional
        The path to save the figure, by default None. If save is True, the save_path
        must be specified.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    # 3d figure
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    slices = np.unique(vcut_planes[:, 0])
    for iSlice in range(slices.size):
        if iSlice % skip != 0:
            continue
        coordinate = vcut_planes[vcut_planes[:, 0] == slices[iSlice], -3:]
        ax.plot(coordinate[:, 0], coordinate[:, 1], coordinate[:, 2], marker, markersize=marker_size)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    if save:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    return None