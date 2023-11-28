import pandas as pd
import sklearn.metrics
import numpy as np


def compute_framewise_ade(pred, target):
    """
    Return ADE for each frame and agent.
    pred [TxAxF]
    trgt [TxAxF]
    """

    ade = np.linalg.norm(pred - target, axis=2)

    return ade

def get_eval_dict_from(raw_data_dict, model_type="reconstruction"):
    """
    Given the prediction dictionary, perform evaluation for the "reconstruction" or
    "one-class" anomaly detectin protocol.
    """

    # for reconstruction methods use the ADE to compute anomaly scores
    if model_type == "reconstruction":

        for seg_id, segment in raw_data_dict.items():

            # input and reconstruction trajectory
            trgt = segment['trgt']

            # compute ADE and FDE
            ades = []
            fdes = []

            for pred in segment['pred']:
                curr_ade = compute_framewise_ade(pred, trgt)
                curr_fde = curr_ade[-1]

                ades.append(curr_ade)
                fdes.append(curr_fde)

            # log
            raw_data_dict[seg_id]['ade'] = ades
            raw_data_dict[seg_id]['fde'] = fdes

    # Frame-wise anomaly score

    # identify all unique sequences
    all_seq_ids = np.concatenate([segment["seq_id"] for _, segment in raw_data_dict.items()], axis=0)
    unique_seq_ids = np.unique(all_seq_ids)

    # for all sequences identify sequence length
    frame_ids = {}

    # identify unique frames
    for seg_id, seg in raw_data_dict.items():

        for seq_id in unique_seq_ids:

            # all agents are considered to have the same seq_id, which should be the case
            if seg["seq_id"][0] == seq_id:

                if seq_id not in frame_ids.keys():
                    frame_ids[seq_id] = [seg["fid"][:, 0, :]]
                else:
                    frame_ids[seq_id].append(seg["fid"][:, 0, :])

    frame_ids = {k: np.concatenate(v) for k, v in frame_ids.items()}
    unique_frame_ids = {k: np.unique(v) for k, v in frame_ids.items()}

    # initialize eval dictionary
    # evaluation implemented for sliding window segments with stride=1, no skipping, and constant number of agents

    eval_dict = {}

    for seq_id in unique_seq_ids:

        eval_dict[seq_id] = {}

        for f_id in unique_frame_ids[seq_id]:
            eval_dict[seq_id][f_id] = {"pred": [], "gt": {"major_label": [], "minor_label": []}}

    # iterate through segments
    for seg_id, seg in raw_data_dict.items():

        seq_id = int(seg["seq_id"][0])

        t_time_steps = seg["obs"].shape[0]
        n_agents = seg["obs"].shape[1]
        k_samples = len(seg["pred"])

        # iterate through al time steps in segment
        for t in range(t_time_steps):

            f_id = int(seg["fid"][t, 0, 0])  # select frame id

            # Reconstruction method: fill eval dictionary with ade
            if model_type == "reconstruction":
                frame_eval_matrix = np.zeros((k_samples, n_agents))

                for k in range(k_samples):
                    frame_eval_matrix[k, :] = seg["ade"][k][t]

                eval_dict[seq_id][f_id]["pred"].append(frame_eval_matrix)

            # Scoring method: fill eval dictionary with score
            elif model_type == "one-class":

                if "score" not in seg.keys():

                    if not eval_dict[seq_id][f_id]["pred"]:
                        del eval_dict[seq_id][f_id]

                else:

                    frame_eval_vector = seg["score"][t, :, 0]

                    eval_dict[seq_id][f_id]["pred"].append(frame_eval_vector)

            # add global label to frame, i.e. normal vs. normal
            major_frame_label = np.max(seg["labels"][t, :, 0])
            minor_frame_label = np.max(seg["labels"][t, :, 1])
            eval_dict[seq_id][f_id]["gt"]["major_label"].append(major_frame_label)
            eval_dict[seq_id][f_id]["gt"]["minor_label"].append(minor_frame_label)

    # for each frame and agent get all metric values
    for seq_id, seq in eval_dict.items():

        for f_id, frame in seq.items():
            # receive scores
            all_scores = np.stack(frame["pred"])

            # take mean over all segments where agent occurs
            mean_scores = np.mean(all_scores, axis=0)

            # take max over all agents in frame
            frame_score = np.max(mean_scores)

            # add to dictionary
            eval_dict[seq_id][f_id]["pred"] = frame_score
            eval_dict[seq_id][f_id]["gt"]["major_label"] = eval_dict[seq_id][f_id]["gt"]["major_label"][0]
            eval_dict[seq_id][f_id]["gt"]["minor_label"] = eval_dict[seq_id][f_id]["gt"]["minor_label"][0]

    return eval_dict


def get_y_true_y_pred(eval_dict):
    y_true = []
    y_score = []

    for seq_id, seq in eval_dict.items():
        for f_id, frame in seq.items():
            y_true.append(frame["gt"]["major_label"])
            y_score.append(frame["pred"])

    return y_true, y_score

def filter_ignore_regions(y_true, y_score):
    """ Filter ignore regions.
    """

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # with ignore regions, i.e. ignore regions labeled with '2'
    valid_indices = np.where(y_true <= 1)[0]
    y_true = y_true[valid_indices]
    y_score = y_score[valid_indices]

    return y_true, y_score


def ad_roc(y_true, y_score):
    """ Compute ROC-curve.
    """

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)

    return fpr, tpr, thresholds


def ad_auroc(y_true, y_score):
    """ Compute Anomaly Detection (AD) metric AUROC.
    """
    auroc = sklearn.metrics.roc_auc_score(y_true, y_score)

    return auroc


def ad_aupr(y_true, y_score):
    """ Compute Anomaly Detection (AD) metric AUPR-abnormal.
    Consider the abnormal class to be the positive class.
    """

    ap_abnormal = sklearn.metrics.average_precision_score(y_true, y_score)

    return ap_abnormal


def fpr_at_95tpr(tpr, fpr):
    """ Compute Anomaly (AD) detction metric FPR-95%-TPR.
    """
    hit = False
    tpr_95_lb = 0
    tpr_95_ub = 0
    fpr_95_lb = 0
    fpr_95_ub = 0

    for i in range(len(tpr)):
        if tpr[i] > 0.95 and not hit:
            tpr_95_lb = tpr[i - 1]
            tpr_95_ub = tpr[i]
            fpr_95_lb = fpr[i - 1]
            fpr_95_ub = fpr[i]
            hit = True

    s = pd.Series([fpr_95_lb, np.nan, fpr_95_ub], [tpr_95_lb, 0.95, tpr_95_ub])

    s = s.interpolate(method="index")

    return s.iloc[1]
