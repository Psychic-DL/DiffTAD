import os
import json
import pickle
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix

'''
time_steps = 10
'''


class Data_Hanlder(object):

    def __init__(self, dataset_name, columns, time_steps):
        self.time_steps = time_steps
        self.data = pd.read_csv(dataset_name, index_col=0)
        self.columns = columns

        self.data['Class'] = 0
        self.data['Class'] = self.data['result'].apply(lambda x: 1 if x == 'normal' else -1)
        self.data[self.columns] = self.data[self.columns].shift(-1) - self.data[self.columns]
        self.data = self.data.dropna(how='any')
        self.pointer = 0
        self.train = np.array([])
        self.test = np.array([])
        self.test_label = np.array([])

        self.split_fraction = 0.2

    def _process_source_data(self):

        self._data_scale()
        self._data_arrage()
        self._split_save_data()

    def _data_scale(self):

        standscaler = StandardScaler()
        mscaler = MinMaxScaler(feature_range=(0, 1))
        self.data[self.columns] = standscaler.fit_transform(self.data[self.columns])
        self.data[self.columns] = mscaler.fit_transform(self.data[self.columns])

    def _data_arrage(self):

        self.all_data = np.array([])
        self.labels = np.array([])
        d_array = self.data[self.columns].values
        class_array = self.data['Class'].values
        for index in range(self.data.shape[0] - self.time_steps + 1):
            this_array = d_array[index:index + self.time_steps].reshape((-1, self.time_steps, len(self.columns)))
            time_steps_label = class_array[index:index + self.time_steps]
            if np.any(time_steps_label == -1):
                this_label = -1
            else:
                this_label = 1
            if self.all_data.shape[0] == 0:
                self.all_data = this_array
                self.labels = this_label
            else:
                self.all_data = np.concatenate([self.all_data, this_array], axis=0)
                self.labels = np.append(self.labels, this_label)

    def _split_save_data(self):
        normal = self.all_data[self.labels == 1]
        abnormal = self.all_data[self.labels == -1]

        split_no = normal.shape[0] - abnormal.shape[0]

        self.train = normal[:split_no, :]
        self.test = np.concatenate([normal[split_no:, :], abnormal], axis=0)
        self.test_label = np.concatenate([np.ones(normal[split_no:, :].shape[0]), -np.ones(abnormal.shape[0])])
        np.save('dataset/train_5000_shuffle.npy', self.train)
        np.save('dataset/test_5000_shuffle.npy', self.test)
        np.save('dataset/test_label_5000_shuffle.npy', self.test_label)

    def _get_data(self):
        if os.path.exists('dataset/train_5000_shuffle.npy'):
            self.train = np.load('dataset/train_5000_shuffle.npy')
            self.test = np.load('dataset/test_5000_shuffle.npy')
            self.test_label = np.load('dataset/test_label_5000_shuffle.npy')
        if self.train.ndim == 3:
            if self.train.shape[1] == self.time_steps and self.train.shape[2] != len(self.columns):
                return 0
        self._process_source_data()

    def fetch_data(self, batch_size):
        if self.train.shape[0] == 0:
            self._get_data()

        if self.train.shape[0] < batch_size:
            return_train = self.train
        else:
            if (self.pointer + 1) * batch_size >= self.train.shape[0] - 1:
                self.pointer = 0
                return_train = self.train[self.pointer * batch_size:, ]
            else:
                self.pointer = self.pointer + 1
                return_train = self.train[self.pointer * batch_size:(self.pointer + 1) * batch_size, ]
        if return_train.ndim < self.train.ndim:
            return_train = np.expand_dims(return_train, 0)
        return return_train

    def plot_confusion_matrix(self, y_true, y_pred, labels, title):
        cmap = plt.cm.binary
        cm = confusion_matrix(y_true, y_pred)
        tick_marks = np.array(range(len(labels))) + 0.5
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 4), dpi=120)
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        intFlag = 0
        for x_val, y_val in zip(x.flatten(), y.flatten()):

            if (intFlag):
                c = cm[y_val][x_val]
                plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=10, va='center', ha='center')

            else:
                c = cm_normalized[y_val][x_val]
                if (c > 0.01):
                    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
                else:
                    plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=10, va='center', ha='center')
        if (intFlag):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
        else:
            plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels)
        plt.yticks(xlocations, labels)
        plt.ylabel('Index of True Classes')
        plt.xlabel('Index of Predict Classes')
        plt.show()


class MyDataHandler(object):

    def __init__(self, path, time_steps, n_rows=20000, re_read=False, normalized=True, val_split=0.1, shuffle=True,
                 skip_abnormal=True):
        self.time_steps = time_steps
        self.path = path
        self.n_rows = n_rows
        self.train = np.array([])
        self.test = np.array([])
        self.test_label = np.array([])
        self.step = 2
        self.re_read = re_read
        self.pointer = 0
        self.val_pointer = 0
        self.normalized = normalized
        self.val_split = val_split
        self.shuffle = shuffle
        self.skip_abnormal = skip_abnormal

    def _read_origin(self):
        anomaly_ids = open("dataset/anomaly_id").readlines()
        self.anomaly_ids = [line.strip() for line in anomaly_ids]
        self.data = pd.read_csv(self.path)
        np.random.seed(42)
        if self.shuffle:
            np.random.shuffle(self.data.values)

        datas = np.array([])
        labels = []
        print("Begin read from: ", self.path)
        for i in tqdm.tqdm(range(self.n_rows)):
            polyline = self.data['POLYLINE'][i]
            name = self.data["TRIP_ID"][i]
            line_points = json.loads(polyline)
            line_points_df = pd.DataFrame(line_points)
            if len(line_points) < self.time_steps:
                continue
            if self.skip_abnormal and str(name) in self.anomaly_ids:
                continue
            index = 0
            while index <= line_points_df.shape[0] - self.time_steps - 1:
                data_item = line_points_df.iloc[index:index + self.time_steps + 1]
                diff = (data_item.shift(-1) - data_item).dropna(how='any').values
                diff = diff[np.newaxis, :, :]
                diff = diff * 100
                if datas.shape[0] == 0:
                    datas = diff
                else:
                    datas = np.concatenate([datas, diff], axis=0)
                labels.append(1)  # todo:  assume all trajectories are normal
                index += self.step
        self.all_data = datas
        self.labels = np.array(labels)
        print("read data: ", self.all_data.shape)

        self.data['Class'] = 0
        self.pointer = 0
        self.split_fraction = 0.2

    def _process_source_data(self):
        self._read_origin()
        self._data_scale()
        self._split_data()

    def _data_scale(self):
        if self.normalized:
            reshape_data_np = self.all_data.reshape((-1, self.all_data.shape[-1]))  # N_points * dim
            self.stand_scaler = StandardScaler()
            self.m_scaler = MinMaxScaler(feature_range=(0, 1))
            reshape_data_np = self.stand_scaler.fit_transform(reshape_data_np)
            reshape_data_np = self.m_scaler.fit_transform(reshape_data_np)
            self.all_data_scale = reshape_data_np.reshape(self.all_data.shape)
            pickle.dump(self.stand_scaler, open("dataset/stand_scaler.pkl", 'wb'))
            pickle.dump(self.m_scaler, open("dataset/m_scaler.pkl", 'wb'))
        else:
            self.all_data_scale = self.all_data

    def _split_data(self):
        # np.random.shuffle(self.all_data_scale)
        normal = self.all_data_scale[self.labels == 1]
        abnormal = self.all_data_scale[self.labels == -1]

        split_no = int((1 - self.val_split) * normal.shape[0])

        self.train = normal[:split_no, :]
        self.test = np.concatenate([normal[split_no:, :], abnormal], axis=0)
        self.test_label = np.concatenate([np.ones(normal[split_no:, :].shape[0]), -np.ones(abnormal.shape[0])])

    def _save_data(self, name="20000"):
        np.save('dataset/train_{}.npy'.format(name), self.train)
        np.save('dataset/test_{}.npy'.format(name), self.test)
        np.save('dataset/test_label_{}.npy'.format(name), self.test_label)

    def _get_data(self):
        if not self.re_read:
            self.train = np.load('dataset/train_20000.npy')
            self.test = np.load('dataset/train_20000.npy')
            self.test_label = np.load('dataset/train_20000.npy')
        else:
            self._process_source_data()
        return 0

    def fetch_data(self, batch_size):
        if self.train.shape[0] == 0:
            self._get_data()

        if self.train.shape[0] < batch_size:
            return_train = self.train
        else:
            if (self.pointer + 1) * batch_size >= self.train.shape[0] - 1:
                self.pointer = 0
                return_train = self.train[self.pointer * batch_size:, ]
            else:
                self.pointer += 1
                return_train = self.train[self.pointer * batch_size:(self.pointer + 1) * batch_size, ]
        if return_train.ndim < self.train.ndim:
            return_train = np.expand_dims(return_train, 0)
        return return_train

    def fetch_val(self, batch_size):
        if self.test.shape[0] < batch_size:
            return_val = self.test
        else:
            if (self.val_pointer + 1) * batch_size >= self.test.shape[0] - 1:
                self.val_pointer = 0
                return_val = self.test[self.val_pointer * batch_size:, ]
            else:
                self.val_pointer += 1
                return_val = self.test[self.pointer * batch_size:(self.val_pointer + 1) * batch_size, ]
        if return_val.ndim < self.test.ndim:
            return_val = np.expand_dims(return_val, 0)
        return return_val

    def vis(self, n_samples=10):
        data = pd.read_csv(self.path, nrows=n_samples)
        print("Begin read from: ", self.path)
        anomaly_ids = ["137265147262", "137265062662", "137264440662", "137264094362", "137264464262"]
        for i in range(300, n_samples):
            polyline = data['POLYLINE'][i]
            name = data["TRIP_ID"][i]
            line_points = json.loads(polyline)
            line_points_df = pd.DataFrame(line_points)
            # if not str(name) in anomaly_ids:
            #     continue
            if len(line_points) < self.time_steps:
                continue

            def on_key_press(event):
                if event.key == "a":
                    print(name)
                plt.close()
            fig = plt.figure()
            plt.title(name)
            plt.plot(line_points_df.values[:, 0], line_points_df.values[:, 1])
            plt.scatter(line_points_df.values[:, 0], line_points_df.values[:, 1])
            fig.canvas.mpl_connect('key_press_event', on_key_press)
            plt.show()
