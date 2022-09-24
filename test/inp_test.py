import meshio
import numpy as np

def test_elset(tmp_path):
    points = np.array(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.5, 0.0], [0.0, 0.5, 0.0]]
    )
    cells = [
        ("triangle", np.array([[0, 1, 2]])),
        ("triangle", np.array([[0, 1, 3]])),
    ]
    cell_sets = {
        "right": [np.array([0]), np.array([])],
        "left": [np.array([]), np.array([1])],
    }
    mesh_ref = meshio.Mesh(points, cells, cell_sets=cell_sets)

    filepath = tmp_path + "test.inp"
    meshio.abaqus.write(filepath, mesh_ref)
    mesh = meshio.abaqus.read(filepath)

    assert np.allclose(mesh_ref.points, mesh.points)

    assert len(mesh_ref.cells) == len(mesh.cells)
    for ic, cell in enumerate(mesh_ref.cells):
        assert cell.type == mesh.cells[ic].type
        assert np.allclose(cell.data, mesh.cells[ic].data)

    assert sorted(mesh_ref.cell_sets.keys()) == sorted(mesh.cell_sets.keys())
    for k, v in mesh_ref.cell_sets.items():
        for ic in range(len(mesh_ref.cells)):
            assert np.allclose(v[ic], mesh.cell_sets[k][ic])

test_elset("./")
