import os
import math
import sys
import datetime
import pickle

import torch
import numpy as np

import networkx as nx


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def file_id_from_path(path):
    import re

    file_name = os.path.split(path)[1]

    digits_in_filename = re.findall("\d", file_name)

    str_id = ""
    for d in digits_in_filename:
        str_id += d

    return int(str_id)


def get_current_time():
    now = datetime.datetime.now()  # current date and time

    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")

    return date_time


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True, adj_type="relative"):
    """
    Define graph structure and compute the adjacency matrix given by adj_type.
    """
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1  # self-aggregation
            for k in range(h + 1, len(step_)):
                if adj_type == "relative":
                    l2_norm = anorm(step_rel[h], step_rel[k])
                elif adj_type == "absolute":
                    l2_norm = anorm(step_[h], step_[k])
                elif adj_type == "identity":
                    l2_norm = 0
                else:
                    sys.exit("Unkonwn adjacency type {}, choose from 'relative' or 'absolute'.".format(adj_type))
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:  # normalize laplacian matrix
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), \
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def load_prediction_data(path):

    with open(path, 'rb') as fin:
        data = pickle.load(fin)

    return data

def get_latent_features(data_dict):

    # extract features
    X = []  # MxF
    seg_shapes = []
    for sid, segment in data_dict.items():
        features = segment["features"]

        n_time_steps, n_agents, n_features = features.shape

        X.append(features.reshape(n_time_steps*n_agents, n_features))
        seg_shapes.append((n_time_steps, n_agents, n_features))

    X = np.concatenate(X)

    return X, seg_shapes

def add_scores_to_data_dict(data_dict, segment_shapes, scores):
    seg_cnt = 0
    start_row = 0
    for sid, segment in data_dict.items():

        n_time_steps, n_agents, n_features = segment_shapes[seg_cnt]

        end_row = start_row + (n_time_steps * n_agents)

        seg_scores = scores[start_row:end_row]

        seg_scores = seg_scores.reshape(n_time_steps, n_agents)

        seg_scores = seg_scores[:, :, np.newaxis]

        segment["score"] = seg_scores

        seg_cnt += 1
        start_row = end_row

    return data_dict

def seq_to_nodes(seq_, max_nodes=88):
    if seq_.shape[0] == 1:
        seq_ = seq_[0]
    else:
        seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]

    return V.squeeze()


def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :], axis=0) + init_node[ped, :]
    return nodes_
