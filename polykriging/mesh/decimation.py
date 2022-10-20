# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import pyvista as pv
import numpy as np
import itertools, time, vtk, multiprocessing


def get_cells(mesh):
    """
    Returns a list of the cells from this mesh with mixed cell types.
    This properly unpacks the VTK cells array.(safe but now so fast)

    Parameters
    ----------
    mesh: pyvista mesh object
        A pyvista mesh object.

    Returns
    -------
    cells: list
        A list of cells
    """
    offset = 0
    cells = []
    for i in range(mesh.n_cells):
        loc = i + offset
        nc = mesh.cells[loc]
        offset += nc
        cell = mesh.cells[loc + 1:loc + nc + 1]
        cells.append(cell)
    return np.array(cells, dtype=np.int32)


def get_edges_from_tetra(cells):
    """
    Given cells of tetrahedral mesh, return all the edges.

    Parameters
    ----------
    cells: A numpy array in the shape of [n_cells, 4]
        The cells array containing node connectivity.

    Returns
    -------
    edges_for_cell: A list of edges
        The edges are sorted so that the first node is always smaller
        than the second. This is to ensure easy searching neighbors.
        The edges of a cell can be retrieved by edges[cell_index]
    edges: A numpy array in the shape of [n_edges, 2]
        The edges array containing node connectivity.
    """
    cells = np.sort(cells, axis=1)

    e1, e2, e3, e4, e5, e6 = \
        [cells[:, i] for i in
         list(itertools.combinations([0, 1, 2, 3], 2))]
    edges_for_cell = np.hstack(
        (e1, e2, e3, e4, e5, e6)
    ).reshape((-1, 6, 2))
    edges = np.unique(edges_for_cell.reshape((-1, 2)), axis=0)
    return edges, edges_for_cell


def get_edge_length(points, edges):
    """
    Returns the length of an edge given node position and the edge.

    Parameters
    ----------
    points: A numpy array in the shape of [n_points, 3]
        The points array containing node position
    edges: A numpy array in the shape of [n_edges, 2]
        The edges array containing node connectivity

    Returns
    -------
    edge_length: A numpy array in the shape of [n_edges]
    """
    return np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)


def adjacent_from_edge(cells, edges, cell_idx=None, return_dict={}):
    """
    Returns the adjacent cells of the edges with the index of the edge as key.

    Parameters
    ----------
    cells: (n, 4) array
        cell list of the mesh expressed in node connectivity
    edges: (m, 2) array
        edge list to be collapsed
    cell_idx: (n,) array
        cell index. If None, it will be generated as np.arange(n). (default: None)

    Returns
    -------
    return_dict: dictionary
        a dictionary of adjacent cells with key as the edge
    """
    if cell_idx is None:
        cell_idx = np.arange(cells.shape[0])

    for edge in edges:
        mask1 = np.any(cells == edge[0], axis=1)
        temp = cell_idx[mask1]
        mask2 = np.any(cells[mask1] == edge[1], axis=1)
        return_dict[str(edge)] = temp[mask2]
    return return_dict


def adjacent_from_edge_parallel(cells, edge_collapse, n_cores=4):
    """
    Get the adjacent cells of the edges to be collapsed.

    Parameters
    ----------
    cells: (n, 4) array
        cell list of the mesh expressed in node connectivity
    edge_collapse: (m, 2) array
        edge list to be collapsed
    n_cores: int
        number of cores to use for multiprocessing (default: 4)

    Returns
    -------
    return_dict: a dictionary of adjacent cells with key as the edge
    """
    cell_idx = np.arange(cells.shape[0])

    print("Creating adjacent info from edges...")
    n_edges = edge_collapse.shape[0]
    n_per_core = int(n_edges / n_cores) + 1
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(n_cores):
        p = multiprocessing.Process(
            target=adjacent_from_edge,
            args=(cells, edge_collapse[i * n_per_core:(i + 1) * n_per_core],
                  cell_idx, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    print("    Done: Adjacent cells from edges are created.")
    return return_dict


def get_boundary_points(mesh):
    """
    Returns a list of boundary points from this mesh.

    Parameters
    ----------
    mesh: pyvista mesh object
    boundary_points: A list of boundary points

    Returns
    -------
    pts_boundary_idx: NumPy array
        A numpy array in the shape of [n_boundary_points] containing the index of boundary points.
    """
    pts_boundary_idx = []
    points = mesh.points
    boundary = mesh.extract_surface()
    pts_boundary = np.array(boundary.points)

    for pts in pts_boundary:
        distance = np.linalg.norm(points - pts, axis=1)
        idx = np.where((distance < 1e-5))[0][0]

        pts_boundary_idx.append(idx)
    return pts_boundary_idx


def get_maximal_independent_node(edges):
    """
    Get the maximal independent node set from the edge list.

    Parameters
    ----------
    edges: (n, 2) array
        edge list

    Returns
    -------
    pts_independent: (m,) array
        maximal independent node set
    """
    import networkx as nx

    G = nx.Graph([tuple(i) for i in edges])
    pts_independent = np.array(nx.maximal_independent_set(G))
    return pts_independent


def get_vertex_indicator(n_points, pts_boundary_idx, pts_independent, edges):
    """
    Get the indicator function of the vertices:
        0 for boundary vertices,
        1 for independent vertices,
        2 for free vertices.
    The indicator function is used to determine the nodes to be collapsed.
    The nodes to be collapsed are the ones with indicator function equal to 2
    while 0 and 1 are fixed.

    Parameters
    ----------
    n_points: int
        number of vertices
    pts_boundary_idx: (m,) array
        indices of boundary points
    pts_independent: (n,) array
        indices of independent points
    edges: (n', 2) array
        edge list

    Returns
    -------
    vertex_indicator: (n_points,) array
        indicator function of the vertices
    edge_indicator: (n', 2) array
        indicator function of the two vertices of an edge
    """
    # Eliminate the boundary vertices contained in pts_independent
    pts_independent = np.setdiff1d(pts_independent, pts_boundary_idx)

    vertex_indicator = np.array([2] * n_points)
    vertex_indicator[pts_boundary_idx] = 0
    vertex_indicator[pts_independent] = 1

    edge_indicator = np.array(vertex_indicator[edges])
    return vertex_indicator, edge_indicator


def get_collapse_direction(edge_indicator):
    """
    Get the direction of edge collapse.
    The direction is determined by the indicator function of the two vertices.

    Parameters
    ----------
    edge_indicator: (n, 2) array
        indicator function of the two vertices of an edge

    Returns
    -------
    collapse_indicator: (n,) array
        direction of edge collapse. possible values are "forward", "backward",
        "bilateral", and "neither"
    """
    collapse_indicator = np.empty(edge_indicator.shape[0], dtype='<U9')

    collapse_indicator[np.all(edge_indicator == [0, 0], axis=1)] = "neither"

    collapse_indicator[np.all(edge_indicator == [0, 1], axis=1)] = "backwards"
    collapse_indicator[np.all(edge_indicator == [1, 0], axis=1)] = "forwards"

    collapse_indicator[np.all(edge_indicator == [0, 2], axis=1)] = "backwards"
    collapse_indicator[np.all(edge_indicator == [2, 0], axis=1)] = "forwards"

    collapse_indicator[np.all(edge_indicator == [1, 2], axis=1)] = "backwards"
    collapse_indicator[np.all(edge_indicator == [2, 1], axis=1)] = "forwards"

    collapse_indicator[np.all(edge_indicator == [2, 2], axis=1)] = "bilateral"

    return collapse_indicator


def get_surf_dist(surf, points):
    """
    Get the distance from a point to the surface mesh by finding the closest
    point on the surface mesh with KDTree.

    Parameters
    ----------
    surf: A pyvista triangular mesh object
    points: (n, 3) array
        points to be measured

    Returns
    -------
    surf_dist: (n,) array of float
        distance from the points to the surface mesh
    idx: (n,) array of int
        index of the closest point on the surface mesh
    """
    from scipy.spatial import KDTree

    tree = KDTree(surf.points)
    surf_dist, idx = tree.query(points)
    # normalize surface distance surf_dist
    surf_dist = (surf_dist - np.min(surf_dist)) / \
                (np.max(surf_dist) - np.min(surf_dist))
    return surf_dist, idx


def get_edge_collapse(points, edges, surf_dist, edge_indicator, threshold=1.2):
    """
    Get the edge collapse list.

    Parameters
    ----------
    points: (n, 3) array
        vertex list
    edges: (n', 2) array
        edge list
    surf_dist: (n,) array
        surface distance of the vertices to interfaces
    edge_indicator: (n', 2) array
        indicator function of the two vertices of an edge. possible values are
        0, 1, and 2 for each containing boundary, independent, and free vertices
        respectively.
    threshold: float
    # TODO: explain this parameter
        threshold for the surface distance to filter out the edges to be collapsed.
        The edges to be collapsed are the ones with edge length less than the threshold
        and surface distance of the two vertices are both greater than the threshold.

    Returns
    -------
    edge_collapse: (n'', 2) array
        edge collapse list
    collapse_indicator: (n'', 2) array
        indicator function of the two vertices of an edge to be collapsed. possible values are
        "forward", "backward", "bilateral", and "neither"
    """
    mask1 = np.unique(edges[:, 0], axis=0, return_index=True)[1]
    edge_collapse = edges[mask1]
    mask2 = np.unique(edge_collapse[:, 1], axis=0, return_index=True)[1]
    edge_collapse = edge_collapse[mask2]

    collapse_indicator = get_collapse_direction(edge_indicator)[mask1][mask2]

    edge_length = get_edge_length(points, edge_collapse)
    mask = edge_length > np.average(surf_dist[edge_collapse], axis=1) * \
           np.max(edge_length) * threshold + np.min(get_edge_length(points, edges))
    collapse_indicator[mask] = "neither"

    for i, node in enumerate(edge_collapse[0]):
        if node in edge_collapse[1]:
            edge_collapse = np.delete(edge_collapse, i, axis=0)

    print("    Number of edges to collapse: {}".format(np.sum(collapse_indicator != "neither")))

    return collapse_indicator, edge_collapse


def renumber_points(pts_del, cells, proc_num, return_dict={}):
    """
    Renumber the points in cells after some points are deleted.

    Parameters
    ----------
    pts_del: A list of points to be deleted
    cells: A numpy array in the shape of [n_cells, 4]
        The cells array containing node connectivity
    proc_num: Int, the number of process for multiprocessing.
    return_dict: A dictionary to store the result of each process.

    Returns
    -------
    num_diff: A numpy array in the shape of [n_cells, 4]
        The difference between the original node index and the new node index
    """
    num_diff = np.zeros(cells.shape, dtype=np.int32)
    for i, pt in enumerate(pts_del):
        mask = cells >= pt
        num_diff[mask] = num_diff[mask] - 1
    return_dict[proc_num] = num_diff
    return num_diff


def tetra_edge_collapse(edges, collapse_indicator, edge_adjacent, points, cells, n_cores):
    """
    Collapse edges of tetrahedral mesh.
    
    Parameters
    ----------
    edges: (n, 2) array

    collapse_indicator: (n,) array
        direction of edge collapse. possible values are "forward", "backward",
        "bilateral", and "neither"
    edge_adjacent: dict
        adjacent cells of each edge
    points: (n, 3) array
        node positions
    cells: (n, 4) array
        node connectivity

    Returns
    -------
    points: (n, 3) array
        node positions after edge collapse
    cells: (n, 4) array
        node connectivity after edge collapse
    """
    print("Edge collapse...")
    pts_del = []
    cell_del = []
    n_points = points.shape[0]

    import copy
    new_points = []
    new_cells = copy.deepcopy(cells)

    if collapse_indicator.shape[0] != edges.shape[0]:
        raise ValueError("collapse_indicator should have the same length as edges.")

    mask = collapse_indicator == "neither"
    collapse_indicator = collapse_indicator[~mask]
    edges = edges[~mask]

    n = 0
    for idx, indicator in np.ndenumerate(collapse_indicator):
        idx = idx[0]
        edge = edges[idx]

        pts_del.append(edges[idx])
        cell_del.append(edge_adjacent[str(edges[idx])])

        if indicator == "forwards":
            start = edges[idx, 0]
            end = edges[idx, 1]
        elif indicator == "backwards":
            start = edges[idx, 1]
            end = edges[idx, 0]
        elif indicator == "bilateral":
            # TODO: implement bilateral collapse. Now it is forwards collapse
            start = edges[idx, 0]
            end = edges[idx, 1]

        new_points.append(points[end, :])
        new_cells[new_cells == start] = n_points + n
        new_cells[new_cells == end] = n_points + n
        n += 1

    # remove unused point and renumber the nodes
    pts_del = np.unique(list(itertools.chain(*pts_del)))

    ### single process version
    # print("size of pts_del", pts_del.shape)
    # print(pts_del)
    # for i, pt in enumerate(pts_del):
    #     mask = new_cells >= pt - i
    #     # new_cells[new_cells >= pt - i] = new_cells[new_cells >= pt - i] - 1
    #     new_cells[mask] = new_cells[mask] - 1

    ### parallel version
    n_pts_del = len(pts_del)
    num_proc = int(n_pts_del / n_cores) + 1

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(n_cores):
        p = multiprocessing.Process(
            target=renumber_points,
            args=(pts_del[i * num_proc:(i + 1) * num_proc],
                  new_cells, i, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    for i in range(1, n_cores):
        return_dict[0] += return_dict[i]
    new_cells += return_dict[0]

    new_points = np.vstack((points, new_points)).astype(np.float64)
    new_points = np.delete(new_points, pts_del, axis=0)

    # remove degenerate cells
    cell_del = np.unique(list(itertools.chain(*cell_del)))
    new_cells = np.delete(new_cells, cell_del, axis=0)

    print("    Done: Edge collapse is done.")

    return new_points, new_cells


def edge_collapse_pipeline(mesh, surf, iteration=1, threshold=2, n_cores=4):
    """
    Edge collapse pipeline. Edges containing boundary and independent points will not be collapsed.

    Parameters
    ----------
    points: (n, 3) array
        vertices
    cells: (m, 4) array
        connectivity
    surf: a surface mesh of interfaces between materials (subdomains)
    iteration: int
        number of iterations for edge collapse
    threshold: float
        threshold for edge collapse

    Returns
    -------
    new_points: (n', 3) array
        vertices after edge collapse
    new_cells: (m', 4) array
        connectivity after edge collapse
    """
    global collapse_indicator, edge_collapse, edge_adjacent, points, cells
    print("Collapsing pipeline starts...")
    points = np.array(mesh.points)
    cells = mesh.cells.reshape((-1, 5))[:, 1:]
    n_points = points.shape[0]

    # Extract all edges from tetrahedral mesh
    print("Tetra mesh edge extraction...")
    edges, edges_for_cell = get_edges_from_tetra(cells)

    pts_independent = get_maximal_independent_node(edges)
    pts_boundary_idx = get_boundary_points(mesh)

    vertex_indicator, edge_indicator = \
        get_vertex_indicator(n_points, pts_boundary_idx, pts_independent, edges)

    # get distance to interfaces
    surf_dist, idx = get_surf_dist(surf, points)

    # get edges to be collapsed
    print("Getting edges to be collapsed...")
    collapse_indicator, edge_collapse = \
        get_edge_collapse(points, edges, surf_dist, edge_indicator, threshold=threshold)

    """adjacent cells for each edge to be collapsed (will be removed later)"""
    s_adj = time.time()
    edge_adjacent = adjacent_from_edge_parallel(cells, edge_collapse, n_cores=n_cores)
    print("    Adjacent cells time: {} seconds".format(time.time() - s_adj))

    s_collapse = time.time()
    new_points, new_cells = tetra_edge_collapse(edge_collapse, collapse_indicator,
                                                edge_adjacent, points, cells, n_cores)
    print("    Edge collapse time: {} seconds".format(time.time() - s_collapse))

    while iteration - 1 > 0:
        grid = construct_tetra_vtk(new_points, new_cells)
        new_points, new_cells = edge_collapse_pipeline(grid, surf, iteration=1, threshold=threshold)
        iteration -= 1
    return new_points, new_cells


def construct_tetra_vtk(points, cells, save=False, filename="tetra.vtk", path="./", binary=True):
    """
    Construct a UnstructuredGrid tetrahedral mesh from vertices and connectivity.

    Parameters
    ----------
    points: (n, 3) array
        vertices
    cells: (m, 4) array
        connectivity
    save: bool
        whether to save the mesh
    filename: str
        if save=True, provide a file name
    path: str
        if save=True, provide a path to save the mesh
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
    if save:
        grid.save(path + filename, binary=binary)
    return grid


if __name__ == "__main__":
    """Read the mesh"""
    mesh = pv.read("./cylinder_ascii.1.vtk")
    surf = mesh.extract_geometry()  # extract surface mesh

    new_points, new_cells = edge_collapse_pipeline(mesh, surf, iteration=3, n_cores=8, threshold=2)

    filename = "cylinder_ascii.2.vtk"
    grid = construct_tetra_vtk(new_points, new_cells, save=True, filename=filename)
