import json
import pywt
import pandas as pd
import matplotlib.pyplot as plt


def wavelet_denoising(data):
    db4 = pywt.Wavelet('db6')
    data = pd.read_csv("/Users/huaqiang.fhq/code/t2vec/data/porto.3000.csv")
    time_steps = 25
    for i in range(data.shape[0]):
        polyline = data['POLYLINE'][i]
        name = data["TRIP_ID"][i]
        line_points = json.loads(polyline)
        line_points_df = pd.DataFrame(line_points)
        if len(line_points) < time_steps:
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
    if type(data) is not None:
        coeffs = pywt.wavedec(data, db4)
        # 高频系数置零
        coeffs[len(coeffs) - 1] *= 0
        meta_1 = pywt.waverec(coeffs, db4)
        coeffs[len(coeffs) - 2] *= 0
        meta_2 = pywt.waverec(coeffs, db4)
        coeffs[len(coeffs) - 3] *= 0
        meta_3 = pywt.waverec(coeffs, db4)
        coeffs[len(coeffs) - 4] *= 0
        meta_4 = pywt.waverec(coeffs, db4)
        return meta_1, meta_2, meta_3, meta_4
