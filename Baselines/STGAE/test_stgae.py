import copy
import os
import argparse
import pickle
import time
import yaml
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.distributions.multivariate_normal
from torch.utils.data import DataLoader

import src.dataset
import src.evaluation
import src.utils
import models.stgae


def test(model, loader_test, KSTEPS=20, export_features=True, deterministic=False):
    model.eval()
    raw_data_dict = {}
    step = 0
    for batch in tqdm(loader_test):
        step += 1

        # Batch
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, obs_traj_rel, frame_ids, seq_ids, labels, V_obs, A_obs = batch

        # Forward
        # V_obs = batch, time, agents, feat
        # V_obs_tmp = batch, feat, time, agents
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        A_obs_in = A_obs[0]  # remove batch dim
        latent_features = None

        if export_features:
            V_recon, _, latent_features = model(V_obs_tmp, A_obs_in)
        else:
            V_recon, _ = model(V_obs_tmp, A_obs_in)

        # Postprocess
        # in: [batch, features, time, agents]
        # out: [time, agents, features]
        V_recon = V_recon.permute(0, 2, 3, 1)
        V_recon = V_recon[0]  # remove batch dim

        V_tr = V_obs
        V_tr = V_tr[0]  # remove batch dim
        num_of_objs = obs_traj_rel.shape[1]
        V_recon, V_tr = V_recon[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        # Logging
        V_x = src.utils.seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_obs = V_obs.data.cpu().numpy()[0]
        V_x_rel_to_abs = src.utils.nodes_rel_to_nodes_abs(V_obs.copy(), V_x[0, :, :].copy())
        V_y_rel_to_abs = V_x_rel_to_abs

        raw_data_dict[step] = {}
        raw_data_dict[step]['fid'] = copy.deepcopy(np.transpose(frame_ids.cpu().numpy()[0], (2, 0, 1)))
        raw_data_dict[step]['seq_id'] = copy.deepcopy(seq_ids.cpu().numpy()[0])
        raw_data_dict[step]['labels'] = copy.deepcopy(np.transpose(labels.cpu().numpy()[0], (2, 0, 1)))
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        if export_features:
            raw_data_dict[step]['features'] = copy.deepcopy(np.transpose(latent_features.cpu().numpy()[0], (0, 2, 1)))
        raw_data_dict[step]['pred'] = []

        if deterministic:  # no sampling

            # mean
            mean = V_recon[:, :, :]
            V_pred = mean

            # relative to absolute coordinates
            V_pred_rel_to_abs = src.utils.nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                                   V_x[0, :, :].copy())

            # log predicted trajectory
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            # log mean and covariance
            raw_data_dict[step]['mean'] = copy.deepcopy(mean.cpu().detach().numpy())
            raw_data_dict[step]['cov'] = ""

        else:  # sampling
            # bi-variate distribution
            mean = V_recon[:, :, :2]
            sx = torch.exp(V_recon[:, :, 2])  # sx
            sy = torch.exp(V_recon[:, :, 3])  # sy
            corr = torch.tanh(V_recon[:, :, 4])  # corr

            epsilon = 1e-5
            cov = torch.zeros(V_recon.shape[0], V_recon.shape[1], 2, 2).cuda()
            cov[:, :, 0, 0] = sx * sx
            cov[:, :, 0, 1] = corr * sx * sy
            cov[:, :, 1, 0] = corr * sx * sy
            cov[:, :, 1, 1] = sy * sy
            I = torch.eye(2).cuda() * epsilon
            cov = cov + I  # to ensure cholesky stability

            mvnormal = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)

            # log mean and covariance
            raw_data_dict[step]['mean'] = copy.deepcopy(mean.cpu().detach().numpy())
            raw_data_dict[step]['cov'] = copy.deepcopy(cov.cpu().detach().numpy())

            # sample KSTEPS times
            for k in range(KSTEPS):

                # sample
                V_pred = mvnormal.sample()

                # relative to absolute coordinates
                V_pred_rel_to_abs = src.utils.nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().copy(),
                                                                       V_x[0, :, :].copy())

                # log prediction
                raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

                for n in range(num_of_objs):
                    pred = []
                    target = []
                    obsrvs = []
                    number_of = []
                    pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                    target.append(V_y_rel_to_abs[:, n:n + 1, :])
                    obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                    number_of.append(1)

    return raw_data_dict


if __name__ == "__main__":

    ### CONFIG ###

    # config file
    parser = argparse.ArgumentParser(description="Test STGAE model.")
    parser.add_argument('--config', type=str, default="config_stgae_test.yaml")

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
                                        adj_type=config_train["model"]["adj_type"])

    loader_test = DataLoader(dset_test,
                             batch_size=1,  # This is irrelative to the args batch size parameter
                             shuffle=False,
                             num_workers=1)

    ### MODEL
    # STGAE model
    model_dict = {"bivariate": {"out_features": 5, "deterministic": False},
                  "mse": {"out_features": 2, "deterministic": True}}
    model_loss_type = config_train["model"]["loss"]
    deterministic = model_dict[model_loss_type]["deterministic"]
    out_features = model_dict[model_loss_type]["out_features"]

    model = models.stgae.stgae(n_stgcnn=config_train["model"]["n_stgcnn"], n_cnn=config_train["model"]["n_cnn"],
                               output_feat=out_features, seq_len=config_train["model"]["obs_len"],
                               kernel_size=config_train["model"]["stgcnn_kernel"],
                               output_encoder_features=True).cuda()

    # load network weights
    model.load_state_dict(torch.load(checkpoint_path))

    ### PREDICTION
    print("\nPredicting...")
    pred_start = time.time()
    raw_prediction_data = test(model, loader_test, deterministic=deterministic, export_features=True)
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
