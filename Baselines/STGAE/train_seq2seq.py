import os
import yaml
import pickle
import argparse
import shutil
import time
import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.utils
import src.dataset
import models.seq2seq


if __name__ == "__main__":

    ### CONFIG ###

    # config file
    parser = argparse.ArgumentParser(description="Train Seq2seq model.")
    parser.add_argument('--config', type=str, default="config_seq2seq_train.yaml")

    args = parser.parse_args()

    ### END CONFIG ###

    # experiment start time
    start_t = time.time()

    ### PATHS & CONFIG
    project_root = os.getcwd()
    data_root = os.path.join(project_root, "datasets/maad")
    exp_root = os.path.join(project_root, "experiments")
    config_root = os.path.join(project_root, "config")

    # config
    config_path = os.path.join(config_root, args.config)
    with open(config_path, "r") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    # experiment
    run_name = "seq2seq_lstm"
    date_time = src.utils.get_current_time()
    run_name = date_time + "_" + run_name
    exp_dir = os.path.join(exp_root, run_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # data
    data_path_train = os.path.join(data_root, config["dataset"]["set"])

    # copy config
    copy_config_dst = os.path.join(exp_dir, "config_train.yaml")
    shutil.copy(config_path, copy_config_dst)


    ### DATA
    dset_train = src.dataset.MAADDataset(data_path_train, obs_len=config["model"]["obs_len"],
                                         adj_type="identity")

    loader_train = torch.utils.data.DataLoader(dset_train,
                                               batch_size=config["train"]["batch_size"],
                                               shuffle=True,
                                               num_workers=0)

    ### MODEL
    # seq2seq model
    model = models.seq2seq.Seq2Seq_LSTM(config["model"]["obs_len"], config["model"]["hidden_dim"],
                                        config["model"]["n_layers"])

    loss_func = nn.MSELoss(reduction='mean')

    # optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=config["optimiser"]["lr"])

    ### TRAIN
    print('\nData and model loaded...\n')
    print('* ' * 30)
    print("Training {}...".format(run_name))
    print('* ' * 30)

    metrics = {'loss': []}

    for epoch in range(1, config["train"]["num_epochs"] + 1):

        model.train()

        train_loss = 0

        for cnt, batch in enumerate(loader_train):

            # get data
            obs_traj, obs_traj_rel, frame_ids, seq_ids, labels, V_obs, A_obs = batch

            optimiser.zero_grad()

            # forward each agent
            reconstructions = []
            N = obs_traj_rel.shape[1]
            for i in range(N):
                reconstructions.append(model(obs_traj_rel[:, i, :, :])[0])

            # compute loss over all agents
            loss = 0
            for i, recon in enumerate(reconstructions):
                loss += loss_func(recon, obs_traj_rel[:, i, :, :])

            # backward
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        train_loss = train_loss / len(loader_train)
        metrics["loss"].append(train_loss)

        print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', train_loss)

        # save model
        torch.save(model.state_dict(), os.path.join(exp_dir, 'epoch_{:03}.pth'.format(epoch)))

        # save metrics
        with open(os.path.join(exp_dir, '00_metrics.pkl'), 'wb') as fp:
            pickle.dump(metrics, fp)

    ### EXPORT & FINISH
    print('* ' * 30 + '\n' + '* ' * 30)
    print("\nTraining summary...")

    # end time
    end_t = time.time()
    print("\nTraining took {}.".format(str(datetime.timedelta(seconds=end_t - start_t))))

    # visualize loss
    train_loss_fig = plt.figure(figsize=(16, 10))
    plt.grid()
    plt.xlabel("Epochs [-]")
    plt.ylabel("Loss [-]")
    plt.plot(metrics["loss"])
    plt.savefig(os.path.join(exp_dir, "00_loss.png"))
    plt.close(train_loss_fig)

    # end training
    print("Training of {} done.".format(run_name))
