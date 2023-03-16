import pickle

from keras.engine.saving import load_model
from config.configutil import getpath
import numpy as np

from util.mathutils import get_rms
from util.viewutil import viewpkresult, viewdeeplearnHis

# trained_path = getpath("wavelet_0_1")
trained_path = getpath("trained_path")

preproced_path = getpath('preproced_path')
data = np.load(preproced_path + "preproced_data_skip_3_hours.npz")

model = load_model(trained_path + "cnn\8\\skip_3_hours.h5")
with open(trained_path + "cnn\8\\skip_3_hours.pkl", 'rb') as hist_file:
    history = pickle.load(hist_file)
# 预测
test_feature = data["test_feature"]
test_label = data["test_label"]
test_feature = test_feature.reshape(test_feature.shape[0], 14, 32, 1)
predict_x = model.predict(test_feature)
predict_x = predict_x.flatten()

viewdeeplearnHis(history, "skip_3_hours_cnn_his")

# 可视化结果
viewpkresult(test_label[:, 0], test_label[:, 1], predict_x, 'cnn')

print(get_rms((test_label[:, 1] - predict_x) * 1000))
