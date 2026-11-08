"""Independent linear 3D Timoshenko-beam check, NOT the authors' ae108 code.

Rigid beam joints, circular sections, periodic displacement fluctuations and
six engineering-strain load cases. Validate against reference labels before use.
"""

from itertools import combinations, product

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree


def beam_matrix(axis, radius, young=1.0, poisson=0.3):
    length = np.linalg.norm(axis)
    if length < 1e-10:
        raise ValueError("Zero-length beam")
    area, inertia = np.pi*radius**2, np.pi*radius**4/4
    shear = young/(2*(1+poisson))
    correction = 6*(1+poisson)/(7+6*poisson)  # Cowper factor for a solid circle.
    local = np.zeros((12, 12))
    pair = np.array([[1., -1.], [-1., 1.]])
    local[np.ix_([0, 6], [0, 6])] = young*area/length*pair
    local[np.ix_([3, 9], [3, 9])] = shear*(2*inertia)/length*pair
    phi = 12*young*inertia/(correction*shear*area*length**2)
    bending = young*inertia/(length**3*(1+phi))*np.array([
        [12, 6*length, -12, 6*length],
        [6*length, (4+phi)*length**2, -6*length, (2-phi)*length**2],
        [-12, -6*length, 12, -6*length],
        [6*length, (2-phi)*length**2, -6*length, (4+phi)*length**2],
    ])
    local[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] = bending
    signs = np.array([1, -1, 1, -1])
    local[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] = bending*signs[:, None]*signs[None, :]
    ex = axis/length
    ey = np.cross(np.eye(3)[np.argmin(abs(ex))], ex)
    ey /= np.linalg.norm(ey)
    frame = np.array([ex, ey, np.cross(ex, ey)])
    transform = np.kron(np.eye(4), frame)
    return transform.T@local@transform


def split_intersections(positions, edges, tol=1e-8):
    """Split crossings and collinear overlaps into welded, unique segments."""
    positions = np.asarray(positions, float)
    edges = np.asarray(edges, int)
    starts = positions[edges[:, 0]]
    vectors = positions[edges[:, 1]]-starts
    params = [[0., 1.] for _ in edges]
    for i, j in combinations(range(len(edges)), 2):
        u, v, difference = vectors[i], vectors[j], starts[j]-starts[i]
        if np.linalg.norm(np.cross(u, v)) <= tol*np.linalg.norm(u)*np.linalg.norm(v):
            if np.linalg.norm(np.cross(u, difference)) > tol*np.linalg.norm(u):
                continue
            for point in positions[edges[j]]:
                t = (point-starts[i])@u/(u@u)
                if -tol <= t <= 1+tol:
                    params[i].append(float(np.clip(t, 0, 1)))
            for point in positions[edges[i]]:
                t = (point-starts[j])@v/(v@v)
                if -tol <= t <= 1+tol:
                    params[j].append(float(np.clip(t, 0, 1)))
        else:
            matrix = np.column_stack([u, -v])
            t, s = np.linalg.lstsq(matrix, difference, rcond=None)[0]
            if (-tol <= t <= 1+tol and -tol <= s <= 1+tol and
                    np.linalg.norm(matrix@np.array([t, s])-difference) < tol):
                params[i].append(float(np.clip(t, 0, 1)))
                params[j].append(float(np.clip(s, 0, 1)))
    # Merge by geometric distance, not rounded-coordinate buckets: two estimates
    # of one intersection can fall on opposite sides of a rounding threshold.
    cut_points = [start+np.unique(values)[:, None]*vector
                  for start, vector, values in zip(starts, vectors, params)]
    offsets = np.r_[0, np.cumsum([len(p) for p in cut_points])]
    points = np.concatenate(cut_points)
    parent = np.arange(len(points))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in cKDTree(points).query_pairs(tol, output_type="ndarray"):
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)
    representatives, ids = np.unique([find(i) for i in range(len(points))], return_inverse=True)
    segments = set()
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        segments.update(tuple(sorted((a, b))) for a, b in zip(ids[lo:hi-1], ids[lo+1:hi]) if a != b)
    return points[representatives], np.array(sorted(segments), int)


def full_cell(adjacency, positions, weld_intersections=False):
    """Reflect [0,1]^3 into [-1,1]^3 and merge coincident beam junctions."""
    edges = np.column_stack(np.where(np.triu(adjacency > 0, 1)))
    if len(edges) == 0:
        raise ValueError("Empty structure")
    if weld_intersections:
        points, segments = split_intersections(positions, edges)
    else:
        active, mapping = np.unique(edges, return_inverse=True)
        points, segments = positions[active], mapping.reshape(-1, 2)
    nodes, lookup, beams = [], {}, set()
    for signs in product([-1., 1.], repeat=3):
        mapping = []
        for point in points*np.array(signs):
            key = tuple(np.round(point, 9))
            if key not in lookup:
                lookup[key] = len(nodes)
                nodes.append(point)
            mapping.append(lookup[key])
        for a, b in segments:
            beams.add(tuple(sorted((mapping[a], mapping[b]))))
    nodes, beams = np.array(nodes), np.array(sorted(beams), int)
    endpoints = nodes[beams]
    on_boundary = np.isclose(abs(endpoints[:, 0]), 1., atol=1e-8) & np.isclose(
        endpoints[:, 0], endpoints[:, 1], atol=1e-8)
    weights = 0.5**on_boundary.sum(axis=1)
    return nodes, beams, weights


def homogenize(adjacency, positions, density=0.15, young=1.0, poisson=0.3, weld_intersections=False, mesh=None):
    nodes, beams, weights = full_cell(adjacency, positions, weld_intersections) if mesh is None else mesh
    length = np.linalg.norm(nodes[beams[:, 1]]-nodes[beams[:, 0]], axis=1)
    volume = 8.0
    radius = np.sqrt(density*volume/(np.pi*(weights@length)))
    periodic = nodes.copy()
    periodic[np.isclose(periodic, 1., atol=1e-8)] = -1.
    _, group = np.unique(np.round(periodic, 9), axis=0, return_inverse=True)
    ndof = 6*(group.max()+1)
    strains = np.zeros((6, 3, 3))
    for i in range(3):
        strains[i, i, i] = 1.
    for mode, (i, j) in enumerate([(1, 2), (2, 0), (0, 1)], 3):
        strains[mode, i, j] = strains[mode, j, i] = 0.5
    affine = np.zeros((len(nodes), 6, 6))
    affine[:, :3, :] = np.einsum("kij,nj->nik", strains, nodes)
    rows, columns, values = [], [], []
    rhs, direct = np.zeros((ndof, 6)), np.zeros((6, 6))
    for (a, b), weight in zip(beams, weights):
        stiffness = weight*beam_matrix(nodes[b]-nodes[a], radius, young, poisson)
        dofs = np.r_[6*group[a]+np.arange(6), 6*group[b]+np.arange(6)]
        prescribed = np.vstack([affine[a], affine[b]])
        rows.extend(np.repeat(dofs, 12))
        columns.extend(np.tile(dofs, 12))
        values.extend(stiffness.ravel())
        np.add.at(rhs, dofs, stiffness@prescribed)
        direct += prescribed.T@stiffness@prescribed
    matrix = sparse.coo_matrix((values, (rows, columns)), shape=(ndof, ndof)).tocsc()
    # Remove only the three translation null modes; no artificial stiffness.
    free = np.arange(3, ndof)
    relaxed = splu(matrix[free][:, free]).solve(-rhs[free])
    residual = np.linalg.norm(matrix[free][:, free]@relaxed+rhs[free])/max(np.linalg.norm(rhs[free]), 1e-15)
    c = (direct+rhs[free].T@relaxed)/volume
    c = (c+c.T)/2
    if not np.isfinite(c).all() or residual > 1e-6:
        raise ValueError("Failed periodic equilibrium solve")
    vector = c[[0, 0, 0, 1, 1, 2, 3, 4, 5], [0, 1, 2, 1, 2, 2, 3, 4, 5]]
    return vector, {"nodes": len(nodes), "beams": len(beams), "periodic_dofs": int(ndof),
                    "radius": float(radius), "equilibrium_relative_residual": float(residual),
                    "minimum_stiffness_eigenvalue": float(np.linalg.eigvalsh(c).min()),
                    "density": density, "young": young, "poisson": poisson,
                    "c_matrix": c.tolist()}


def analytic_checks():
    radius, length, poisson = 0.03, 1., 0.3
    area, inertia = np.pi*radius**2, np.pi*radius**4/4
    shear, correction = 1/(2*(1+poisson)), 6*(1+poisson)/(7+6*poisson)
    stiffness = beam_matrix(np.array([length, 0., 0.]), radius)
    flexibility = np.linalg.inv(stiffness[6:, 6:])
    expected = [length/area, length**3/(3*inertia)+length/(correction*shear*area),
                length/(shear*2*inertia)]
    actual = [flexibility[0, 0], flexibility[1, 1], flexibility[3, 3]]
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    for omega in np.eye(3):
        rigid = np.r_[np.zeros(3), omega, np.cross(omega, [length, 0., 0.]), omega]
        np.testing.assert_allclose(stiffness@rigid, 0., atol=1e-12)
    points = np.array(list(product([0., 0.5, 1.], repeat=3)))
    distances = np.linalg.norm(points[:, None]-points[None, :], axis=2)
    adjacency = np.isclose(distances, 0.5)
    c, _ = homogenize(adjacency, points)
    np.testing.assert_allclose(c[[0, 3, 5]], 0.15/3, rtol=1e-9)
    c2, _ = homogenize(adjacency, points, young=2.)
    np.testing.assert_allclose(c2, 2*c, rtol=1e-9, atol=1e-12)
    return {"cantilever_axial_bending_torsion": True, "beam_rigid_modes": True,
            "periodic_cubic_axial_stiffness": True, "uniform_modulus_scaling": True}
