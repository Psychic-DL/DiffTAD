import os
import time
import yaml
import pickle
import shutil
import datetime
import argparse
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim

import src.utils
import src.loss
import src.dataset
import models.stgae


def graph_loss(V_pred ,V_target):
    return src.loss.bivariate_loss(V_pred, V_target)

def train(loader_train, epoch, loss_func, exp_dir, batch_size=128):
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len / batch_size) * batch_size + loader_len % batch_size - 1

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, obs_traj_rel, frame_ids, seq_ids, labels, V_obs, A_obs = batch

        # define reconstruction target
        V_tr = V_obs

        optimiser.zero_grad()

        # Forward
        # V_obs = batch, seq, node, feat
        # V_obs_tmp = batch, feat, seq, node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.permute(0, 2, 3, 1)

        # prepare target
        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % batch_size != 0 and cnt != turn_point:
            l = loss_func(V_pred, V_tr)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / batch_size
            is_fst_loss = True
            loss.backward()
            optimiser.step()
            # Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    # save model
    torch.save(model.state_dict(), os.path.join(exp_dir, "epoch_{:03}.pth".format(epoch)))

    # return last loss in epoch
    return loss_batch / batch_count

if __name__ == "__main__":

    ### CONFIG ###

    # config file
    parser = argparse.ArgumentParser(description="Train STGAE model.")
    parser.add_argument('--config', type=str, default="config_stgae_train.yaml")

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
    run_name = "stgae_" + config["model"]["loss"] + "_loss"
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
                                         adj_type=config["model"]["adj_type"])

    loader_train = torch.utils.data.DataLoader(dset_train,
                                               batch_size=1,  # This is irrelative to the batch size parameter
                                               shuffle=True,
                                               num_workers=0)

    ### MODEL
    # STGAE model
    model_dict = {"bivariate": {"out_features": 5, "loss": graph_loss},
                  "mse": {"out_features": 2, "loss": nn.MSELoss()}}
    model_loss_type = config["model"]["loss"]
    out_features = model_dict[model_loss_type]["out_features"]
    loss_func = model_dict[model_loss_type]["loss"]

    model = models.stgae.stgae(n_stgcnn=config["model"]["n_stgcnn"], n_cnn=config["model"]["n_cnn"],
                               output_feat=out_features, seq_len=config["model"]["obs_len"],
                               kernel_size=config["model"]["stgcnn_kernel"]).cuda()

    # optimiser
    optimiser = torch.optim.SGD(model.parameters(), lr=config["optimiser"]["init_lr"])

    if config["optimiser"]["use_lrschd"]:
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser,
                                              step_size=config["optimiser"]["lr_sh_rate"],
                                              gamma=config["optimiser"]["lr_sh_gamma"])

    ### TRAIN
    print('\nData and model loaded...\n')
    print('* '*30)
    print("Training {}...".format(run_name))
    print('* '*30)

    metrics = {'loss': []}

    for epoch in range(1, config["train"]["num_epochs"]+1):

        # train one epoch
        current_loss = train(loader_train, epoch, loss_func, exp_dir=exp_dir, batch_size=config["train"]["batch_size"])
        metrics["loss"].append(current_loss)

        # update learning reate scheduler
        if config["optimiser"]["use_lrschd"]:
            scheduler.step()

        # save metrics
        with open(os.path.join(exp_dir, '00_metrics.pkl'), 'wb') as fp:
            pickle.dump(metrics, fp)

    ### EXPORT & FINISH
    print('* '*30 + '\n' + '* '*30)
    print("\nTraining summary...")

    # end time
    end_t = time.time()
    print("\nTraining took {}.".format(str(datetime.timedelta(seconds=end_t-start_t))))

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
