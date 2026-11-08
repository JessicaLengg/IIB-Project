"""Selected upstream functions, retained with their original AST bodies.

Copyright (c) 2023 Li Zheng. MIT License; see LICENSE.
Source: https://github.com/li-zhengz/UnifyingTrussDesignSpace
"""
from pathlib import Path
import numpy as np
from scipy import spatial

DATA = Path(__file__).resolve().parent / "data"
nodesInit = np.loadtxt(DATA / "nodes_init.csv", delimiter=",")
numNodes, dim, dist = 27, 3, 0.5
tree = spatial.KDTree(nodesInit)
rmin = np.sqrt(3) / 2 + 1e-6
x_edge_nodes = [1, 7, 19, 25]
y_edge_nodes = [3, 5, 21, 23]
z_edge_nodes = [9, 11, 15, 17]
xy_face_nodes = [4, 22]
yz_face_nodes = [12, 14]
xz_face_nodes = [10, 16]
body_nodes = [13]
dof_nodes = (x_edge_nodes + y_edge_nodes + z_edge_nodes + xy_face_nodes
             + yz_face_nodes + xz_face_nodes + body_nodes)


def str2stru(x):
    if x == 'A1':
        a = 1
        stack = 'sc'
        s = ''
    elif x == 'A2':
        a = 2
        stack = 'sc'
        s = '_S'
    elif x == 'B2':
        a = 0
        stack = 'bcc'
        s = '_S'
    elif x == 'C2':
        a = 3
        stack = 'fcc'
        s = '_S'
    return (a, stack, s)

def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin) * np.random.rand(n) + vmin

def moveTop(adj, x, uc_idx):
    if uc_idx == 0:
        to_move_id = np.array([17])
    elif uc_idx == 1:
        to_move_id = np.array([1, 3, 5, 7, 9, 11, 17, 15, 19, 21, 23, 25])
    elif uc_idx == 2:
        to_move_id = np.array([17, 21, 25])
    elif uc_idx == 3:
        to_move_id = np.array([14, 10, 22, 16, 12, 4])
    m_adj = adj.copy()
    m_x = x.copy()
    modify_uc = 1
    if modify_uc == 1:
        for k in to_move_id:
            a = tree.query_ball_point(nodesInit[k, :], rmin)
            neighbor_idx = np.array(a)
            to_move = np.random.choice([0, 1], 1)
            if to_move == 1:
                m_adj[k, :] = 0.0
                m_adj[:, k] = 0.0
                selected_id = np.random.choice(neighbor_idx, 1)
                m_adj[k, selected_id] = 1.0
                m_adj[selected_id, k] = 1.0
        invalid_node_id = np.where(np.sum(m_adj, axis=0) == 1.0)[0]
        new_ex_adj = m_adj.copy()
        iterk = 0
        while iterk >= 0:
            for j in invalid_node_id:
                a = tree.query_ball_point(nodesInit[j, :], rmin)
                neighbor_idx = np.array(a)
                selected_id = np.random.choice(neighbor_idx, 1)
                to_move = np.random.choice([0, 1], 1)
                if to_move == 1:
                    new_ex_adj[j, selected_id] = 1.0
                    new_ex_adj[selected_id, j] = 1.0
                else:
                    new_ex_adj[j, :] = 0.0
                    new_ex_adj[:, j] = 0.0
                for k in range(numNodes):
                    new_ex_adj[k, k] = 0.0
            invalid_node_id = np.where(np.sum(new_ex_adj, axis=0) == 1.0)[0]
            if invalid_node_id.shape[0] > 0:
                iterk += 1
            else:
                break
    else:
        new_ex_adj = m_adj
    return new_ex_adj

def check_bc_connected(adj, x, numUC):
    not_on_boundary = []
    for i in range(numUC):
        ex_adj = adj[i * numNodes:(i + 1) * numNodes, :]
        ex_x = x[i * numNodes:(i + 1) * numNodes, :]
        ex_adj += ex_adj.transpose()
        row = np.nonzero(ex_adj)[0]
        col = np.nonzero(ex_adj)[0]
        bc_i = np.max(ex_x[row, 1])
        bc_j = np.max(ex_x[col, 1])
        if bc_i < 1.0 and bc_j < 1.0:
            not_on_boundary.append(i)
        bc_i = np.min(ex_x[row, 1])
        bc_j = np.min(ex_x[col, 1])
        if bc_i > 0.0 and bc_j > 0.0:
            not_on_boundary.append(i)
        bc_i = np.min(ex_x[row, -1])
        bc_j = np.min(ex_x[col, -1])
        if bc_i > 0.0 and bc_j > 0.0:
            not_on_boundary.append(i)
        bc_i = np.max(ex_x[row, -1])
        bc_j = np.max(ex_x[col, -1])
        if bc_i < 1.0 and bc_j < 1.0:
            not_on_boundary.append(i)
        bc_i = np.max(ex_x[row, 0])
        bc_j = np.max(ex_x[col, 0])
        if bc_i < 1.0 and bc_j < 1.0:
            not_on_boundary.append(i)
        bc_i = np.min(ex_x[row, 0])
        bc_j = np.min(ex_x[col, 0])
        if bc_i > 0.0 and bc_j > 0.0:
            not_on_boundary.append(i)
    return np.array(not_on_boundary)

def check_connection(adj, x, numUC):
    for i in range(numUC):
        ex_adj = adj[i * numNodes:(i + 1) * numNodes, :]
        ex_x = x[i * numNodes:(i + 1) * numNodes, :].copy()
        ex_adj += ex_adj.transpose()
        invalid_node = np.where(np.sum(ex_adj, axis=0) == 1.0)[0]
        for k in invalid_node:
            if k == 0:
                if (ex_x[k, 0] == 0.0) | (ex_x[k, 1] == 0.0) | (ex_x[k, 2] == 0.0) is True:
                    pass
                else:
                    ex_x[k, np.random.choice([0, 1, 2], 1)] = 0.0
            elif k == 2:
                if (ex_x[k, 0] == 1.0) | (ex_x[k, 1] == 0.0) | (ex_x[k, 2] == 0.0) is True:
                    pass
                else:
                    idx = np.random.choice([0, 1, 2], 1)
                    if idx == 0:
                        ex_x[k, idx] = 1.0
                    else:
                        ex_x[k, idx] = 0.0
            elif k == 6:
                if (ex_x[k, 0] == 0.0) | (ex_x[k, 1] == 1.0) | (ex_x[k, 2] == 0.0) is True:
                    pass
                else:
                    idx = np.random.choice([0, 1, 2], 1)
                    if idx == 1:
                        ex_x[k, idx] = 1.0
                    else:
                        ex_x[k, idx] = 0.0
            elif k == 8:
                if (ex_x[k, 0] == 1.0) | (ex_x[k, 1] == 1.0) | (ex_x[k, 2] == 0.0) is True:
                    pass
                else:
                    idx = np.random.choice([0, 1, 2], 1)
                    if idx == 2:
                        ex_x[k, idx] = 0.0
                    else:
                        ex_x[k, idx] = 1.0
            elif k == 26:
                if (ex_x[k, 0] == 0.0) | (ex_x[k, 1] == 0.0) | (ex_x[k, 2] == 1.0) is True:
                    pass
                else:
                    idx = np.random.choice([0, 1, 2], 1)
                    if idx == 2:
                        ex_x[k, idx] = 1.0
                    else:
                        ex_x[k, idx] = 0.0
            elif k == 28:
                if (ex_x[k, 0] == 1.0) | (ex_x[k, 1] == 0.0) | (ex_x[k, 2] == 1.0) is True:
                    pass
                else:
                    idx = np.random.choice([0, 1, 2], 1)
                    if idx == 1:
                        ex_x[k, idx] = 0.0
                    else:
                        ex_x[k, idx] = 1.0
            elif k == 32:
                if (ex_x[k, 0] == 0.0) | (ex_x[k, 1] == 1.0) | (ex_x[k, 2] == 1.0) is True:
                    pass
                else:
                    idx = np.random.choice([0, 1, 2], 1)
                    if idx == 0:
                        ex_x[k, idx] = 0.0
                    else:
                        ex_x[k, idx] = 1.0
            elif k == 34:
                if (ex_x[k, 0] == 1.0) | (ex_x[k, 1] == 1.0) | (ex_x[k, 2] == 1.0) is True:
                    pass
                else:
                    ex_x[k, np.random.choice([0, 1, 2], 1)] = 1.0
        x[i * numNodes:(i + 1) * numNodes, :] = ex_x.copy()
    return (adj, x)

def get_dof_idx(node_number):
    i = node_number
    if i in x_edge_nodes:
        dof_idx = [3 * i]
    elif i in y_edge_nodes:
        dof_idx = [3 * i + 1]
    elif i in z_edge_nodes:
        dof_idx = [3 * i + 2]
    elif i in xy_face_nodes:
        dof_idx = [3 * i + 0, 3 * i + 1]
    elif i in yz_face_nodes:
        dof_idx = [3 * i + 1, 3 * i + 2]
    elif i in xz_face_nodes:
        dof_idx = [3 * i + 0, 3 * i + 2]
    elif i in body_nodes:
        dof_idx = [3 * i + 0, 3 * i + 1, 3 * i + 2]
    return dof_idx

def randomPtb(j, ptb_last_iter):
    ptb_vec = ptb_last_iter.flatten()
    dof_idx = get_dof_idx(j)
    ptb_tmp = randrange(len(dof_idx), -dist, dist)
    for j in range(len(dof_idx)):
        k = dof_idx[j]
        ptb_vec[k] += ptb_tmp[j]
    ptb = ptb_vec.reshape([numNodes, dim])
    return ptb

def ptbNodes(adj):
    non_zero_idx = np.where(np.sum(adj, axis=0) != 0.0)[0]
    ptb_iter = np.zeros([numNodes, dim])
    for j in non_zero_idx:
        if j in dof_nodes:
            to_move = np.random.choice([0, 1], 1)
            if to_move == 1:
                ptb_next_iter = randomPtb(j, ptb_iter)
                ptb_iter = ptb_next_iter.copy()
    return ptb_iter

def bary_coor1D(r):
    return r + 0.5

def bary_coor2D(s, t):
    m1, n1 = (1, 1)
    m2, n2 = (1, 0)
    m3, n3 = (0, 0)
    m4, n4 = (0, 1)
    N1 = 1.0 / 4 * (1 - s) * (1 - t)
    N2 = 1.0 / 4 * (1 - s) * (1 + t)
    N3 = 1.0 / 4 * (1 + s) * (1 + t)
    N4 = 1.0 / 4 * (1 + s) * (1 - t)
    ms = N1 * m1 + N2 * m2 + N3 * m3 + N4 * m4
    ns = N1 * n1 + N2 * n2 + N3 * n3 + N4 * n4
    return (ms, ns)

def bary_coor3D(s, t, q):
    x1, y1, z1 = (1, 1, 1)
    x2, y2, z2 = (1, 1, 0)
    x3, y3, z3 = (1, 0, 1)
    x4, y4, z4 = (1, 0, 0)
    x5, y5, z5 = (0, 1, 1)
    x6, y6, z6 = (0, 1, 0)
    x7, y7, z7 = (0, 0, 1)
    x8, y8, z8 = (0, 0, 0)
    N1 = 1.0 / 8 * (1 - s) * (1 - t) * (1 - q)
    N2 = 1.0 / 8 * (1 - s) * (1 - t) * (1 + q)
    N3 = 1.0 / 8 * (1 - s) * (1 + t) * (1 - q)
    N4 = 1.0 / 8 * (1 - s) * (1 + t) * (1 + q)
    N5 = 1.0 / 8 * (1 + s) * (1 - t) * (1 - q)
    N6 = 1.0 / 8 * (1 + s) * (1 - t) * (1 + q)
    N7 = 1.0 / 8 * (1 + s) * (1 + t) * (1 - q)
    N8 = 1.0 / 8 * (1 + s) * (1 + t) * (1 + q)
    xs = N1 * x1 + N2 * x2 + N3 * x3 + N4 * x4 + N5 * x5 + N6 * x6 + N7 * x7 + N8 * x8
    ys = N1 * y1 + N2 * y2 + N3 * y3 + N4 * y4 + N5 * y5 + N6 * y6 + N7 * y7 + N8 * y8
    zs = N1 * z1 + N2 * z2 + N3 * z3 + N4 * z4 + N5 * z5 + N6 * z6 + N7 * z7 + N8 * z8
    return (xs, ys, zs)

def ptb2nodes(new_ptb):
    nodes_n = nodesInit.copy()
    for n in x_edge_nodes:
        nodes_n[n, 0] = bary_coor1D(new_ptb[n, 0])
    for n in y_edge_nodes:
        nodes_n[n, 1] = bary_coor1D(new_ptb[n, 1])
    for n in z_edge_nodes:
        nodes_n[n, 2] = bary_coor1D(new_ptb[n, 2])
    for n in xy_face_nodes:
        s, t = (new_ptb[n, 0], new_ptb[n, 1])
        nodes_n[n, 0], nodes_n[n, 1] = bary_coor2D(s, t)
    for n in yz_face_nodes:
        s, t = (new_ptb[n, 1], new_ptb[n, 2])
        nodes_n[n, 1], nodes_n[n, 2] = bary_coor2D(s, t)
    for n in xz_face_nodes:
        s, t = (new_ptb[n, 0], new_ptb[n, 2])
        nodes_n[n, 0], nodes_n[n, 2] = bary_coor2D(s, t)
    for n in body_nodes:
        s, t, q = (new_ptb[n, 0], new_ptb[n, 1], new_ptb[n, 2])
        nodes_n[n, 0], nodes_n[n, 1], nodes_n[n, 2] = bary_coor3D(s, t, q)
    return nodes_n
