import os
import math
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import src.utils


class MAADDataset(Dataset):
    """
    Dataloder for the multi-agent anomaly detection (MAAD) dataset.
    """

    def __init__(
            self, data_dir, obs_len=15, skip=1, min_agents=0, delim='\t', adj_type="relative"):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <agent_id> <x> <y> <behavior> <subclass>
        - obs_len: Number of time-steps in input trajectories
        - skip: Number of frames to skip in the sliding window segmentation
        - min_agents: Minimum number of agents that should be in a sequence
        - delim: Delimiter in the dataset file
        - adj_type: Type of adjacency matrix, "relative" vs. "absolute" vs. "identity"
        """
        super(MAADDataset, self).__init__()

        self.max_agents_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.skip = skip
        self.seq_len = self.obs_len
        self.delim = delim
        self.adj_type = adj_type

        # data files
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        # init data lists
        num_agents_in_seq = []
        seg_list = []
        seg_list_rel = []
        frame_ids_list = []
        seq_id_list = []
        labels_list = []

        # load sequences
        print("\nLoading dataset...")

        for seq_id, path in enumerate(tqdm(all_files)):
            data = src.utils.read_file(path, delim)

            # organize sequence data per frame
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
                num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            # sliding window segmentation
            for idx in range(0, num_sequences * self.skip + 1, skip):

                # get segment data
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                agents_in_curr_seq = np.unique(curr_seq_data[:, 2])
                self.max_agents_in_frame = max(self.max_agents_in_frame, len(agents_in_curr_seq))

                # initialize segment
                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(agents_in_curr_seq), 2, self.seq_len))
                curr_frame_ids = np.zeros((len(agents_in_curr_seq), 1, self.seq_len))
                curr_labels = np.zeros((len(agents_in_curr_seq), 2, self.seq_len))
                curr_seq_id = np.zeros((len(agents_in_curr_seq), 1))

                # iterate through agents in segment
                num_agents_considered = 0
                _non_linear_agent = []
                for _, agent_id in enumerate(agents_in_curr_seq):

                    # get agent data
                    curr_agent_seq = curr_seq_data[curr_seq_data[:, 2] == agent_id, :]
                    curr_agent_seq = np.around(curr_agent_seq, decimals=4)

                    # define padding
                    pad_front = frames.index(curr_agent_seq[0, 0]) - idx
                    pad_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len or curr_agent_seq.shape[0] != self.seq_len:
                        continue

                    # get agent frames
                    curr_agent_frames = curr_agent_seq[:, 0]

                    # get agent labels
                    curr_agent_labels = np.transpose(curr_agent_seq[:, -2:])

                    # get agent trajectory
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 3:5])

                    # log segment data
                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    rel_curr_agent_seq[:, 1:] = curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]
                    _idx = num_agents_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_agent_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_agent_seq
                    curr_frame_ids[_idx, :, pad_front:pad_end] = curr_agent_frames
                    curr_seq_id[_idx] = seq_id
                    curr_labels[_idx, :, pad_front:pad_end] = curr_agent_labels

                    num_agents_considered += 1

                if num_agents_considered > min_agents:
                    num_agents_in_seq.append(num_agents_considered)
                    frame_ids_list.append(curr_frame_ids)
                    seq_id_list.append(curr_seq_id)
                    labels_list.append(curr_labels)
                    seg_list.append(curr_seq[:num_agents_considered])
                    seg_list_rel.append(curr_seq_rel[:num_agents_considered])

        self.num_seq = len(seg_list)
        seg_list = np.concatenate(seg_list, axis=0)
        seg_list_rel = np.concatenate(seg_list_rel, axis=0)
        frame_ids_list = np.concatenate(frame_ids_list, axis=0)
        seq_id_list = np.concatenate(seq_id_list, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        # convert numpy -> torch tensor
        self.obs_traj = torch.from_numpy(seg_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seg_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.frame_ids = torch.from_numpy(frame_ids_list[:, :, :self.obs_len]).type(torch.int)
        self.seq_ids = torch.from_numpy(seq_id_list).type(torch.int)
        self.labels = torch.from_numpy(labels_list[:, :, :self.obs_len]).type(torch.int)
        cum_start_idx = [0] + np.cumsum(num_agents_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        # convert to graphs
        self.v_obs = []
        self.A_obs = []

        print("\nConverting trajectory data to graph format...")

        pbar = tqdm(total=len(self.seq_start_end))

        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_, a_ = src.utils.seq_to_graph(self.obs_traj[start:end, :],
                                            self.obs_traj_rel[start:end, :],
                                            adj_type=self.adj_type)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())

        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :],  # [BxNxFxT]
            self.obs_traj_rel[start:end, :],  # [BxNxFxT]
            self.frame_ids[start:end, :],  # [BxNx1xT]
            self.seq_ids[start:end, :],  # [BxNx1]
            self.labels[start:end, :],  # [BxNx2xT]
            self.v_obs[index],  # [BxTxNxF]
            self.A_obs[index]  # [BxTxNxN]
        ]

        # B: batch size
        # T: sequence length
        # N: number of agents
        # F: number of features

        return out


if __name__ == "__main__":
    ### CONFIGURATION
    set_name = "test"
    obs_seq_len = 15
    skip = 1

    ### DATA LOADING
    project_root = os.path.join(os.getcwd(), "..")
    data_root = os.path.join(project_root, "datasets/maad")
    data_dir = os.path.join(data_root, set_name)

    dataset = MAADDataset(data_dir, obs_len=obs_seq_len, skip=skip)

    print("Done loading trajectory data.")
