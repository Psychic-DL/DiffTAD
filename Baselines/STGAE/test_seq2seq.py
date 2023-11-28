import os
import yaml
import json
import pickle
import argparse
import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import src.utils
import src.dataset
import src.evaluation
import models.seq2seq

if __name__ == "__main__":

    ### CONFIG ###

    # config file
    parser = argparse.ArgumentParser(description="Test Seq2Seq model.")
    parser.add_argument('--config', type=str, default="config_seq2seq_test.yaml")

    args = parser.parse_args()

    ### END CONFIG ###

    ### PATHS & CONFIG
    project_root = os.getcwd()
    data_root = os.path.join(project_root, "datasets/maad")
    exp_root = os.path.join(project_root, "experiments")
    config_root = os.path.join(project_root, "config")

    # load config test
    config_test_path = os.path.join(config_root, args.config)
    with open(config_test_path, "r") as fin:
        config_test = yaml.load(fin, Loader=yaml.FullLoader)

    # run
    run_root = os.path.join(exp_root, config_test["run_name"])

    # checkpoint
    checkpoint_path = os.path.join(run_root, config_test["checkpoint"])

    # load config train
    config_train_path = os.path.join(run_root, "config_train.yaml")
    with open(config_train_path, "r") as fin:
        config_train = yaml.load(fin, Loader=yaml.FullLoader)

    # data
    data_path_test = os.path.join(data_root, config_test["test_set"])

    # create evaluation directory
    eval_dir = "eval_" + src.utils.get_current_time() + "_" + config_test["test_set"]
    eval_path = os.path.join(run_root, eval_dir)

    if not os.path.isdir(eval_path):
        os.makedirs(eval_path)

    ### DATA
    dset_test = src.dataset.MAADDataset(data_path_test, obs_len=config_train["model"]["obs_len"],
                                        adj_type="identity")

    loader_test = DataLoader(dset_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1)

    ### MODEL
    # seq2seq model
    model = models.seq2seq.Seq2Seq_LSTM(config_train["model"]["obs_len"], config_train["model"]["hidden_dim"],
                                        config_train["model"]["n_layers"]).cuda()

    # load network weights
    model.load_state_dict(torch.load(checkpoint_path))

    ### PREDICTION
    print("\nPredicting...")
    pred_start = time.time()

    model.eval()
    raw_prediction_data = {}
    step = 0

    for cnt, batch in enumerate(loader_test):

        step += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, obs_traj_rel, frame_ids, seq_ids, labels, V_obs, A_obs = batch

        # forward each agent
        reconstructions = []
        N = obs_traj_rel.shape[1]
        for i in range(N):
            # single-agent reconstruction [1, F, T]
            recon = model(obs_traj_rel[:, i, :, :])[0].cpu().detach().numpy()
            reconstructions.append(recon)

        # prediction [T, N, F]
        pred = np.concatenate(reconstructions, axis=0)
        pred = np.transpose(pred, (2, 0, 1))

        # process observation and target
        V_x = src.utils.seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_obs = V_obs.data.cpu().numpy()[0]
        V_x_rel_to_abs = src.utils.nodes_rel_to_nodes_abs(V_obs.copy(), V_x[0, :, :].copy())
        V_y_rel_to_abs = V_x_rel_to_abs

        # post-process
        V_x_pred_rel_to_abs = src.utils.nodes_rel_to_nodes_abs(pred, V_x[0, :, :].copy())

        # log
        raw_prediction_data[step] = {}
        raw_prediction_data[step]['fid'] = copy.deepcopy(np.transpose(frame_ids.cpu().numpy()[0], (2, 0, 1)))
        raw_prediction_data[step]['seq_id'] = copy.deepcopy(seq_ids.cpu().numpy()[0])
        raw_prediction_data[step]['labels'] = copy.deepcopy(np.transpose(labels.cpu().numpy()[0], (2, 0, 1)))
        raw_prediction_data[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_prediction_data[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_prediction_data[step]['pred'] = copy.deepcopy(V_x_pred_rel_to_abs)

    pred_end = time.time()

    ### EVALUATION
    print("Evaluating...")

    # create evaluation dictionary
    eval_dict = src.evaluation.get_eval_dict_from(raw_prediction_data, model_type="reconstruction")

    # y_true and y_pred from eval dict
    y_true, y_score = src.evaluation.get_y_true_y_pred(eval_dict)

    # compute metrics only iff more than one classes occurs in y_true
    if not src.utils.all_equal(y_true):

        # filter ignore regions
        y_true, y_score = src.evaluation.filter_ignore_regions(y_true, y_score)

        # ROC
        fpr, tpr, thresholds = src.evaluation.ad_roc(y_true, y_score)

        # AUROC
        auroc = src.evaluation.ad_auroc(y_true, y_score)

        # AUPR-abnormal
        ap_abnormal = src.evaluation.ad_aupr(y_true, y_score)

        # AUPR-normal
        ap_normal = src.evaluation.ad_aupr(1 - y_true, -y_score)

        # FPR @ 95-TPR
        fpr_at_95tpr = src.evaluation.fpr_at_95tpr(tpr, fpr)

        # log results
        results = {"auroc": auroc, "aupr-abnormal": ap_abnormal, "aupr-normal": ap_normal, "fpr @ 95-tpr": fpr_at_95tpr}
        for mname, mval in results.items():
            print("{:20}: {:8.6f}".format(mname, mval))

        # export results
        output_file = open(os.path.join(eval_path, "results.json"), "w")
        json.dump(results, output_file)

    ### EXPORT
    # export raw prediction data
    output_file = open(os.path.join(eval_path, "prediction_data.pkl"), "wb")
    pickle.dump(raw_prediction_data, output_file)
    output_file.close()

    # export eval data
    output_file = open(os.path.join(eval_path, "eval_data.pkl"), "wb")
    pickle.dump(eval_dict, output_file)
    output_file.close()

    print("Done Testing.")
