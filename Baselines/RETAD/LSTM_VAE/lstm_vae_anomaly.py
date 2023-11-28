# -*- coding: utf-8 -*-
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.nn.rnn_cell import MultiRNNCell, LSTMCell
from utils import Data_Hanlder, MyDataHandler


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak * x)


def _LSTMCells(unit_list, act_fn_list, cell_type="GRU"):
    if cell_type == "LSTM":
        return MultiRNNCell([tf.nn.rnn_cell.LSTMCell(unit,
                                                     activation=act_fn)
                             for unit, act_fn in zip(unit_list, act_fn_list)])
    elif cell_type == "GRU":
        return MultiRNNCell([tf.nn.rnn_cell.GRUCell(unit,
                                                    activation=act_fn)
                             for unit, act_fn in zip(unit_list, act_fn_list)])


class LSTM_VAE(object):
    def __init__(self, path_name, input_dim, z_dim, time_steps, outlier_fraction, n_trajectory=2000, pre_trained=False,
                 vis=False):
        self.outlier_fraction = outlier_fraction
        self.n_trajectory = n_trajectory

        self.n_hidden = 16
        self.batch_size = 128
        self.learning_rate = 0.0002
        self.train_iters = 400

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.time_steps = time_steps

        self.pointer = 0
        self.anomaly_score = 0
        self.sess = tf.Session()
        self.data_source = MyDataHandler(path_name, time_steps, self.n_trajectory, re_read=True, val_split=0.5,
                                         normalized=False, shuffle=False)
        self._build_network()
        self.pre_trained = pre_trained
        self.vis = vis
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self):
        with tf.variable_scope('ph'):
            self.X = tf.placeholder(tf.float32, shape=[None, self.time_steps, self.input_dim], name='input_X')

        with tf.variable_scope('encoder'):
            with tf.variable_scope('lat_mu'):
                mu_fw_lstm_cells = _LSTMCells([self.z_dim], [lrelu])
                mu_bw_lstm_cells = _LSTMCells([self.z_dim], [lrelu])

                (mu_fw_outputs, mu_fw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    mu_fw_lstm_cells,
                    mu_bw_lstm_cells,
                    self.X, dtype=tf.float32)
                mu_outputs = tf.add(mu_fw_outputs, mu_fw_outputs)

            with tf.variable_scope('lat_sigma'):
                sigma_fw_lstm_cells = _LSTMCells([self.z_dim], [tf.nn.softplus])
                sigma_bw_lstm_cells = _LSTMCells([self.z_dim], [tf.nn.softplus])
                (sigma_fw_outputs, sigma_bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    sigma_fw_lstm_cells,
                    sigma_bw_lstm_cells,
                    self.X, dtype=tf.float32)
                sigma_outputs = tf.add(sigma_fw_outputs, sigma_bw_outputs)
                sample_Z = mu_outputs + sigma_outputs * tf.random_normal(
                    tf.shape(mu_outputs),
                    0, 1, dtype=tf.float32)

        with tf.variable_scope('decoder'):
            recons_lstm_cells = _LSTMCells([self.n_hidden, self.input_dim], [lrelu, lrelu])
            self.recons_X, _ = tf.nn.dynamic_rnn(recons_lstm_cells, sample_Z, dtype=tf.float32)

        with tf.variable_scope('loss'):
            reduce_dims = np.arange(1, tf.keras.backend.ndim(self.X))
            recons_loss = tf.losses.mean_squared_error(self.X, self.recons_X)
            kl_loss = - 0.5 * tf.reduce_mean(1 + sigma_outputs - tf.square(mu_outputs) - tf.exp(sigma_outputs))
            smooth_loss = tf.reduce_sum(tf.square(self.recons_X))
            max_loss = tf.reduce_max(self.recons_X)
            self.opt_loss = recons_loss + kl_loss + smooth_loss * 0.5 + max_loss * 0.5
            self.all_losses = tf.reduce_sum(tf.square(self.X - self.recons_X), reduction_indices=reduce_dims)

        with tf.variable_scope('train'):
            self.uion_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.opt_loss)

    @staticmethod
    def hausdorff_distance(traj, rec_traj):
        from scipy.spatial.distance import directed_hausdorff
        hs_distance = directed_hausdorff(traj, rec_traj)[0]
        return hs_distance

    def train(self):
        saver = tf.train.Saver()
        if self.pre_trained:
            new_saver = tf.train.import_meta_graph('./checkpoint/MyModel-9000.meta')
            new_saver.restore(self.sess, tf.train.latest_checkpoint('./checkpoint'))
        for i in range(self.train_iters):
            this_X = self.data_source.fetch_data(self.batch_size)
            self.sess.run([self.uion_train_op], feed_dict={
                self.X: this_X
            })

            if i % 200 == 0:
                # val_X = self.data_source.fetch_val(16)
                mse_loss, recons_X = self.sess.run([self.opt_loss, self.recons_X], feed_dict={
                    self.X: self.data_source.test
                })
                print('{}, round {}: with loss: {}'.format(time.asctime(time.localtime(time.time())), i, mse_loss))
            if i % 1000 == 0:
                saver.save(self.sess, "./checkpoint/MyModel", global_step=i)
        self._arange_score(self.data_source.train)

    def eval(self):
        new_saver = tf.train.import_meta_graph('./checkpoint/MyModel-9000.meta')
        # new_saver.restore(self.sess, tf.train.latest_checkpoint('./checkpoint'))

        # graph = tf.get_default_graph()
        # recons_X = graph.get_tensor_by_name("decoder/rnn/transpose_1:0")
        # all_loss = graph.get_tensor_by_name("loss/Sum:0")
        # opt_loss = graph.get_tensor_by_name("loss/add_1:0")
        n_samples = 1000
        data = pd.read_csv("/Users/huaqiang.fhq/code/t2vec/data/porto.csv", nrows=n_samples)
        anomaly_ids = open("dataset/anomaly_id").readlines()
        anomaly_ids = [line.strip() for line in anomaly_ids]
        for i in range(n_samples):

            polyline = data['POLYLINE'][i]
            name = data["TRIP_ID"][i]
            line_points = json.loads(polyline)
            line_points_df = pd.DataFrame(line_points)
            if len(line_points) < self.time_steps:
                continue
            normal_type = str(name) in anomaly_ids
            # if not normal_type:
            #     continue
            index = 0
            datas_batch = np.array([])
            loss_list = []
            h_distance_list = []
            # print(name, normal_type)
            while index <= line_points_df.shape[0] - self.time_steps - 1:
                data_item = line_points_df.iloc[index:index + self.time_steps + 1]
                diff = (data_item.shift(-1) - data_item).dropna(how='any').values
                diff = diff[np.newaxis, :, :]
                diff = diff * 100
                if datas_batch.shape[0] == 0:
                    datas_batch = diff
                else:
                    datas_batch = np.concatenate([datas_batch, diff], axis=0)
                index += 1
                _opt_loss, _recons_X = self.sess.run([self.opt_loss, self.recons_X], feed_dict={
                    self.X: diff
                })
                h_distance = LSTM_VAE.hausdorff_distance(_recons_X[0], diff[0])
                h_distance_list.append(h_distance)

                origin_X = np.cumsum(diff[0], axis=0)
                recons_X = np.cumsum(_recons_X[0], axis=0)

                if self.vis:
                    plt.figure()
                    plt.plot(origin_X[:, 0], origin_X[:, 1], color="blue")
                    plt.plot(recons_X[:, 0], recons_X[:, 1], color="red")
                    plt.title("loss: {} normal type: {}".format(_opt_loss, h_distance, normal_type))
                    plt.show()
                loss_list.append(_opt_loss)
            if len(loss_list) > 0:
                print(name, normal_type, max(loss_list), max(h_distance_list))

    def _arange_score(self, input_data):
        input_all_losses = self.sess.run(self.all_losses, feed_dict={
            self.X: input_data
        })
        self.anomaly_score = np.percentile(input_all_losses, (1 - self.outlier_fraction) * 100)

    def judge(self, test):
        all_test_loss = self.sess.run(self.all_losses, feed_dict={
            self.X: test
        })
        result = map(lambda x: 1 if x < self.anomaly_score else -1, all_test_loss)

        return list(result)

    def plot_confusion_matrix(self):
        predict_label = self.judge(self.data_source.test)
        self.data_source.plot_confusion_matrix(self.data_source.test_label, predict_label, ['Abnormal', 'Normal'],
                                               'LSTM_VAE Confusion-Matrix')


def main():
    lstm_vae = LSTM_VAE('/Users/huaqiang.fhq/code/t2vec/data/porto.3000.csv', 2, z_dim=8, time_steps=24,
                        outlier_fraction=0.01, n_trajectory=500, pre_trained=True)
    # lstm_vae.data_source.vis(n_samples=1000)
    lstm_vae.train()
    lstm_vae.eval()
    # lstm_vae.plot_confusion_matrix()


if __name__ == '__main__':
    main()
