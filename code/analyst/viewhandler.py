import os
import pickle

import joblib
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from dtw import *
import scipy.io as sio
from keras.engine.saving import load_model
from scipy import signal
from wavelets import WaveletAnalysis

from config.configutil import getpath
class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format
def compSciOrder(val):
    irg = range(0,1000)
    n = 0
    for  i in irg:
        if np.power(10,i)<=val and  np.power(10,i+1)>=val:
            n = i
            break
    return n

# show all rtkplot time series
def viewallpreproc():
    preproced_path = getpath('preproced_path')
    wfiles = os.listdir(preproced_path)
    data4origin = getpath('data4origin')
    data4mat = {}
    for file in wfiles:
        if file.endswith('.npz'):
            data = np.load(os.path.join(preproced_path, file))
            t = data["label"][:, 0]
            t = (t - t[0]) * 7 * 24
            north = data["label"][:, 1] * 1000
            east = data["label"][:, 2] * 1000
            up = data["label"][:, 3] * 1000
            print(file, ',', np.std(north), ',', np.std(east), ',', np.std(up))
            plt.plot(t, east)
            out4origin = np.zeros((data['label'].shape[0], 4))
            out4origin[:, 0] = t
            out4origin[:, 1] = north
            out4origin[:, 2] = east
            out4origin[:, 3] = up
            np.savetxt(data4origin + file[0:len(file) - 4] + "_raw_neu.txt", out4origin, fmt="%f", delimiter=",")
            data4mat[file[0:len(file) - 4]] = out4origin
    plt.axis([0, 24, -15, 15])
    plt.show()
    sio.savemat(data4origin + "all_raw_neu.mat", data4mat)


def view_cnn_train_hist(cnnworkpath, modelname):
    cnnhistpath = cnnworkpath + 'hist\\'
    cnnhistviewpath = cnnworkpath + 'hist_view\\'
    with open(cnnhistpath + modelname + ".pkl", 'rb') as hist_file:
        history = pickle.load(hist_file)
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='test')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title(modelname)
    plt.savefig(cnnhistviewpath + modelname + '.png')
    plt.show()


def view_common_predict(workpath, predictname, direct = 0):
    outpath = workpath + '\\out\\'
    viewpath = workpath + 'view\\'
    outext = '.npz'
    presult = np.load(outpath + predictname + outext)
    t = presult['t']
    t = (t-t[0])*7*24
    refer = presult['refer']
    predict = presult['predict']
    filtered = presult['filtered']
    # axts = plt.subplot(2,1,1)

    plt.figure()

    rectts = [0.12, 0.6, 0.86, 0.38]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    # rect2 = [0.14, 0.05, 0.77, 0.2]

    axts = plt.axes(rectts)
    # ax2 = plt.axes(rect2)


    axts.plot(t, refer, 'r.')
    axts.plot(t, predict, 'g.')
    axts.grid(True)
    # plt.title(predictname)
    plt.xlabel("Time (hour)")
    if direct == 1:
        plt.ylabel("North (mm)")
    elif direct ==2:
        plt.ylabel("East (mm)")
    elif direct ==3:
        plt.ylabel("Up (mm)")
    else:
        plt.ylabel("Displacement (mm)")
    plt.legend(['original', 'predicted'], loc = 2, ncol = 2)


    # fs = 1 / 30.0
    # plot psd
    # plt.subplot(2,1,2)
    # f, Pper_spec = signal.periodogram(refer, fs, 'flattop', scaling='spectrum')
    # plt.plot(f, Pper_spec,'r.-')
    # f, Pper_spec = signal.periodogram(predict, fs, 'flattop', scaling='spectrum')
    # plt.plot(f, Pper_spec,'g.-')
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD')
    # plt.grid()
    # plt.show()
    # pre_waref=WaveletAnalysis(refer, dt=30.0/3600)
    # scales = pre_waref.scales
    # dj = pre_waref.dj
    # s0 = pre_waref.s0
    # pre_wapred = WaveletAnalysis(predict, dt=30.0/3600)
    # if(pre_waref.scales[-1]< pre_wapred.scales[-1]):
    #     scales = pre_wapred.scales
    #     dj = pre_wapred.dj
    #     s0 = pre_wapred.s0
    waref = WaveletAnalysis(refer, dt=30.0/3600)
    wapred = WaveletAnalysis(predict, dt=30.0/3600)
    # axref = plt.subplot(2, 2, 3)
    rectref = [0.12, 0.09, 0.395, 0.38]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    axref = plt.axes(rectref)

    figref = axref.figure
    # figref.subplots_adjust(wspace=0, hspace=0)
    # wavelet power spectrum
    wpower_ref = waref.wavelet_power
    # scales
    scales = waref.scales
    # associated time vector
    t = waref.time
    T, S = np.meshgrid(t, scales)
    CS = axref.contourf(T, S, wpower_ref, 100)
    n = compSciOrder(np.max(wpower_ref))
    axref.set_yscale('log')
    # cbar = figref.colorbar(CS, format=OOMFormatter(n, mathText=False))
    axref.set_xlabel("Time (hour)")
    axref.set_xticks([0,5,10,15,20])
    axref.set_yscale('log')
    axref.set_ylabel("Period (hour)")
    axref.set_title("original", fontsize='10')
    # axref.set_title('true')
    # cbar.ax.set_ylabel('verbosity coefficient')

    # axpred = plt.subplot(2, 2, 4)
    rectpred = [0.515, 0.09, 0.395, 0.38]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    axpred = plt.axes(rectpred)
    # plt.figure(figsize=(320, 480))
    figpred = axpred.figure
    wpower_pred = wapred.wavelet_power
    ratio = np.max(wpower_pred)/np.max(wpower_pred)
    # scales
    # associated time vector
    t = wapred.time
    T, S = np.meshgrid(t, scales)
    # n = compSciOrder(np.max(wapred))
    # axpred.set_yscale('log')
    axpred.contourf(T, S, wpower_pred/ratio, 100)
    # cbar = figpred.colorbar(CS, format=OOMFormatter(n, mathText=False))
    # figpred.colorbar(CS,ax = [axref,axpred],format=OOMFormatter(n, mathText=False))

    axpred.set_xlabel("Time (hour)")
    axpred.set_yscale('log')
    axpred.set_yticks([])
    axpred.set_xticks([0,5,10,15,20])
    axpred.set_title("predicted", fontsize='10')
    position = figpred.add_axes([0.92, 0.09, 0.015, .38])  # 位置[左,下,右,上]
    cb = figpred.colorbar(CS, cax=position, format=OOMFormatter(n, mathText=False))
    plt.tight_layout()
    # cbar.ax.set_ylabel('verbosity coefficient')
    plt.savefig(viewpath +predictname+'_'+str(direct)+'.tif', bbox_inches='tight', pad_inches=0.0)
    plt.show()


def viewnedpreproc(dir, file):
    data = np.load(os.path.join(dir, file))
    n = data["label"][:, 1] * 1000
    e = data["label"][:, 2] * 1000
    d = data["label"][:, 3] * 1000
    t = data["label"][:, 0]
    plt.plot(t, n, 'r', label='north: %.2f mm' % np.std(n))
    plt.plot(t, e, 'g', label='east: %.2f mm' % np.std(e))
    plt.plot(t, d, 'b', label='down: %.2f mm' % np.std(d))
    plt.legend()
    plt.savefig(getpath('model_view_temp') + file + '_ned.png')
    plt.show()


def viewnedpreprocoverview(dir):
    ns = []
    es = []
    ds = []
    for root, _, wfiles in os.walk(dir):
        for file in wfiles:
            data = np.load(os.path.join(root, file))
            n = data["label"][:, 1] * 1000
            e = data["label"][:, 2] * 1000
            d = data["label"][:, 3] * 1000
            ns.append(np.mean(n))
            es.append(np.mean(e))
            ds.append(np.mean(d))
    plt.plot(ns, 'r', label='north')
    plt.plot(es, 'g', label='east')
    plt.plot(ds, 'b', label='down')
    plt.legend()
    plt.show()


def viewrfrmodel(feature, label, modelname):
    trained_path = getpath("trained_path")
    rfr = joblib.load(trained_path + "rfr\\" + modelname)
    featshp = feature.shape
    predict = rfr.predict(feature.reshape((featshp[0], featshp[1] * featshp[2])))
    predict_north = predict[:, 0] * 1000
    predict_east = predict[:, 1] * 1000
    predict_up = predict[:, 2] * 1000
    refer_north = label[:, 1] * 1000
    refer_east = label[:, 2] * 1000
    refer_up = label[:, 3] * 1000
    t = label[:, 0]
    filter_north = refer_north - predict_north
    filter_east = refer_east - predict_east
    filter_up = refer_up - predict_up
    std_r_north = np.std(refer_north)
    std_r_east = np.std(refer_east)
    std_r_up = np.std(refer_up)
    std_p_north = np.std(predict_north)
    std_p_east = np.std(predict_east)
    std_p_up = np.std(predict_up)
    std_f_north = np.std(filter_north)
    std_f_east = np.std(filter_east)
    std_f_up = np.std(filter_up)
    std_r = np.sqrt(std_r_north * std_r_north + std_r_east * std_r_east + std_r_up * std_r_up)
    std_p = np.sqrt(std_p_north * std_p_north + std_p_east * std_p_east + std_p_up * std_p_up)
    std_f = np.sqrt(std_f_north * std_f_north + std_f_east * std_f_east + std_f_up * std_f_up)
    plt.subplot(3, 1, 1)
    plt.plot(t, refer_north, 'r', label='refer:%.2f mm' % std_r_north)
    plt.plot(t, predict_north, 'g', label='predict:%.2f mm' % std_p_north)
    plt.plot(t, filter_north, 'b', label='filtered:%.2f mm' % std_f_north)
    plt.legend()
    plt.ylabel('north')
    plt.title(modelname)
    plt.subplot(3, 1, 2)
    plt.plot(t, refer_east, 'r', label='refer:%.2f mm' % std_r_east)
    plt.plot(t, predict_east, 'g', label='predict:%.2f mm' % std_p_east)
    plt.plot(t, filter_east, 'b', label='filtered:%.2f mm' % std_f_east)
    plt.legend()
    plt.ylabel('east')
    plt.subplot(3, 1, 3)
    plt.plot(t, refer_up, 'r', label='refer:%.2f mm' % std_r_up)
    plt.plot(t, predict_up, 'g', label='predict:%.2f mm' % std_p_up)
    plt.plot(t, filter_up, 'b', label='filtered:%.2f mm' % std_f_up)
    plt.legend()
    plt.ylabel('up')
    plt.xlabel('time（week)')
    plt.savefig(trained_path + "rfr\\" + modelname + '.png')
    plt.show()
    outputdata = np.zeros((label.shape[0], 12))
    outputdata[:, 0] = label[:, 0]
    outputdata[:, 1] = (label[:, 0] - label[0, 0]) * 7 * 24 * 3600
    outputdata[:, 2] = (label[:, 0] - label[0, 0]) * 7 * 24
    outputdata[:, 3:6] = label[:, 1:4] * 1000
    outputdata[:, 6] = predict_north
    outputdata[:, 7] = predict_east
    outputdata[:, 8] = predict_up
    outputdata[:, 9] = filter_north
    outputdata[:, 10] = filter_east
    outputdata[:, 11] = filter_up
    data4origin = getpath('data4origin')
    print(str(std_r_north), str(std_r_east), str(std_r_up), str(std_p_north), str(std_p_east), str(std_p_up),
          str(std_f_north), str(std_f_east), str(std_f_up), str(std_r), str(std_p), str(std_f))
    np.savetxt(data4origin + "\\bjfs_rfr_neu.txt", outputdata, fmt="%f", delimiter=",")


def viewsvr_rbfmodel(feature, label, modelname):
    trained_path = getpath("trained_path")
    svr_rbf = joblib.load(trained_path + "svr_rbf\\" + modelname)
    featshp = feature.shape
    predict = svr_rbf.predict(feature.reshape((featshp[0], featshp[1] * featshp[2])))
    predict_north = predict[:, 0] * 1000
    predict_east = predict[:, 1] * 1000
    predict_up = predict[:, 2] * 1000
    refer_north = label[:, 1] * 1000
    refer_east = label[:, 2] * 1000
    refer_up = label[:, 3] * 1000
    t = label[:, 0]
    filter_north = refer_north - predict_north
    filter_east = refer_east - predict_east
    filter_up = refer_up - predict_up
    std_r_north = np.std(refer_north)
    std_r_east = np.std(refer_east)
    std_r_up = np.std(refer_up)
    std_p_north = np.std(predict_north)
    std_p_east = np.std(predict_east)
    std_p_up = np.std(predict_up)
    std_f_north = np.std(filter_north)
    std_f_east = np.std(filter_east)
    std_f_up = np.std(filter_up)
    std_r = np.sqrt(std_r_north * std_r_north + std_r_east * std_r_east + std_r_up * std_r_up)
    std_p = np.sqrt(std_p_north * std_p_north + std_p_east * std_p_east + std_p_up * std_p_up)
    std_f = np.sqrt(std_f_north * std_f_north + std_f_east * std_f_east + std_f_up * std_f_up)
    plt.subplot(3, 1, 1)
    plt.plot(t, refer_north, 'r', label='refer:%.2f mm' % std_r_north)
    plt.plot(t, predict_north, 'g', label='predict:%.2f mm' % std_p_north)
    plt.plot(t, filter_north, 'b', label='filtered:%.2f mm' % std_f_north)
    plt.legend()
    plt.ylabel('north')
    plt.title(modelname)
    plt.subplot(3, 1, 2)
    plt.plot(t, refer_east, 'r', label='refer:%.2f mm' % std_r_east)
    plt.plot(t, predict_east, 'g', label='predict:%.2f mm' % std_p_east)
    plt.plot(t, filter_east, 'b', label='filtered:%.2f mm' % std_f_east)
    plt.legend()
    plt.ylabel('east')
    plt.subplot(3, 1, 3)
    plt.plot(t, refer_up, 'r', label='refer:%.2f mm' % std_r_up)
    plt.plot(t, predict_up, 'g', label='predict:%.2f mm' % std_p_up)
    plt.plot(t, filter_up, 'b', label='filtered:%.2f mm' % std_f_up)
    plt.legend()
    plt.ylabel('up')
    plt.xlabel('time（week)')
    plt.savefig(trained_path + "svr_rbf\\" + modelname + '.png')
    plt.show()
    outputdata = np.zeros((label.shape[0], 12))
    outputdata[:, 0] = label[:, 0]
    outputdata[:, 1] = (label[:, 0] - label[0, 0]) * 7 * 24 * 3600
    outputdata[:, 2] = (label[:, 0] - label[0, 0]) * 7 * 24
    outputdata[:, 3:6] = label[:, 1:4] * 1000
    outputdata[:, 6] = predict_north
    outputdata[:, 7] = predict_east
    outputdata[:, 8] = predict_up
    outputdata[:, 9] = filter_north
    outputdata[:, 10] = filter_east
    outputdata[:, 11] = filter_up
    data4origin = getpath('data4origin')
    print(str(std_r_north), str(std_r_east), str(std_r_up), str(std_p_north), str(std_p_east), str(std_p_up),
          str(std_f_north), str(std_f_east), str(std_f_up), str(std_r), str(std_p), str(std_f))
    np.savetxt(data4origin + "\\bjfs_svr_rbf_neu.txt", outputdata, fmt="%f", delimiter=",")


def viewsvr_rbf_eachmodel(feature, label, model, directIdx):
    if directIdx == 1:
        direct = 'north'
    elif directIdx == 2:
        direct = 'east'
    else:
        direct = 'up'
    modelname = direct + '_' + model
    trained_path = getpath("trained_path")
    svr_rbf = joblib.load(trained_path + "svr_rbf\\" + modelname)
    featshp = feature.shape
    predict = svr_rbf.predict(feature.reshape((featshp[0], featshp[1] * featshp[2])))
    predict_val = predict * 1000

    refer = label[:, directIdx] * 1000
    t = label[:, 0]
    filter = refer - predict_val

    std_r = np.std(refer)
    std_p = np.std(predict_val)
    std_f = np.std(filter)
    plt.plot(t, refer, 'r', label='refer:%.2f mm' % std_r)
    plt.plot(t, predict_val, 'g', label='predict:%.2f mm' % std_p)
    plt.plot(t, filter, 'b', label='filtered:%.2f mm' % std_f)
    plt.legend()
    plt.ylabel(direct)
    plt.title(modelname)
    plt.xlabel('time（week)')
    plt.savefig(trained_path + "svr_rbf\\" + modelname + '.png')
    plt.show()
    outputdata = np.zeros((label.shape[0], 6))
    outputdata[:, 0] = label[:, 0]
    outputdata[:, 1] = (label[:, 0] - label[0, 0]) * 7 * 24 * 3600
    outputdata[:, 2] = (label[:, 0] - label[0, 0]) * 7 * 24
    outputdata[:, 3] = label[:, directIdx] * 1000
    outputdata[:, 4] = predict_val
    outputdata[:, 5] = filter

    data4origin = getpath('data4origin')
    np.savetxt(data4origin + "\\" + modelname[0:modelname.index('.')] + ".txt", outputdata, fmt="%f", delimiter=",")


def viewsvr_rfr_eachmodel(feature, label, model, directIdx):
    if directIdx == 1:
        direct = 'north'
    elif directIdx == 2:
        direct = 'east'
    else:
        direct = 'up'
    modelname = direct + '_' + model
    trained_path = getpath("trained_path")
    rfr = joblib.load(trained_path + "erf\\" + modelname)
    featshp = feature.shape
    predict = rfr.predict(feature.reshape((featshp[0], featshp[1] * featshp[2])))
    predict_val = predict * 1000

    refer = label[:, directIdx] * 1000
    t = label[:, 0]
    filter = refer - predict_val

    std_r = np.std(refer)
    std_p = np.std(predict_val)
    std_f = np.std(filter)
    plt.plot(t, refer, 'r', label='refer:%.2f mm' % std_r)
    plt.plot(t, predict_val, 'g', label='predict:%.2f mm' % std_p)
    plt.plot(t, filter, 'b', label='filtered:%.2f mm' % std_f)
    plt.legend()
    plt.ylabel(direct)
    plt.title(modelname)
    plt.xlabel('time（week)')
    plt.savefig(trained_path + "erf\\" + modelname + '.png')
    plt.show()
    outputdata = np.zeros((label.shape[0], 6))
    outputdata[:, 0] = label[:, 0]
    outputdata[:, 1] = (label[:, 0] - label[0, 0]) * 7 * 24 * 3600
    outputdata[:, 2] = (label[:, 0] - label[0, 0]) * 7 * 24
    outputdata[:, 3] = label[:, directIdx] * 1000
    outputdata[:, 4] = predict_val
    outputdata[:, 5] = filter

    data4origin = getpath('data4origin')
    np.savetxt(data4origin + "\\" + modelname[0:modelname.index('.')] + ".txt", outputdata, fmt="%f", delimiter=",")


def viewerfmodel(feature, label, modelname):
    trained_path = getpath("trained_path")
    erf = joblib.load(trained_path + "erf\\" + modelname)
    featshp = feature.shape
    predict = erf.predict(feature.reshape((featshp[0], featshp[1] * featshp[2])))
    predict_north = predict[:, 0] * 1000
    predict_east = predict[:, 1] * 1000
    predict_up = predict[:, 2] * 1000
    refer_north = label[:, 1] * 1000
    refer_east = label[:, 2] * 1000
    refer_up = label[:, 3] * 1000
    t = label[:, 0]
    filter_north = refer_north - predict_north
    filter_east = refer_east - predict_east
    filter_up = refer_up - predict_up
    std_r_north = np.std(refer_north)
    std_r_east = np.std(refer_east)
    std_r_up = np.std(refer_up)
    std_p_north = np.std(predict_north)
    std_p_east = np.std(predict_east)
    std_p_up = np.std(predict_up)
    std_f_north = np.std(filter_north)
    std_f_east = np.std(filter_east)
    std_f_up = np.std(filter_up)
    std_r = np.sqrt(std_r_north * std_r_north + std_r_east * std_r_east + std_r_up * std_r_up)
    std_p = np.sqrt(std_p_north * std_p_north + std_p_east * std_p_east + std_p_up * std_p_up)
    std_f = np.sqrt(std_f_north * std_f_north + std_f_east * std_f_east + std_f_up * std_f_up)
    plt.subplot(3, 1, 1)
    plt.plot(t, refer_north, 'r', label='refer:%.2f mm' % std_r_north)
    plt.plot(t, predict_north, 'g', label='predict:%.2f mm' % std_p_north)
    plt.plot(t, filter_north, 'b', label='filtered:%.2f mm' % std_f_north)
    plt.legend()
    plt.ylabel('north')
    plt.title(modelname)
    plt.subplot(3, 1, 2)
    plt.plot(t, refer_east, 'r', label='refer:%.2f mm' % std_r_east)
    plt.plot(t, predict_east, 'g', label='predict:%.2f mm' % std_p_east)
    plt.plot(t, filter_east, 'b', label='filtered:%.2f mm' % std_f_east)
    plt.legend()
    plt.ylabel('east')
    plt.subplot(3, 1, 3)
    plt.plot(t, refer_up, 'r', label='refer:%.2f mm' % std_r_up)
    plt.plot(t, predict_up, 'g', label='predict:%.2f mm' % std_p_up)
    plt.plot(t, filter_up, 'b', label='filtered:%.2f mm' % std_f_up)
    plt.legend()
    plt.ylabel('up')
    plt.xlabel('time（week)')
    plt.savefig(trained_path + "erf\\" + modelname + '.png')
    plt.show()
    outputdata = np.zeros((label.shape[0], 12))
    outputdata[:, 0] = label[:, 0]
    outputdata[:, 1] = (label[:, 0] - label[0, 0]) * 7 * 24 * 3600
    outputdata[:, 2] = (label[:, 0] - label[0, 0]) * 7 * 24
    outputdata[:, 3:6] = label[:, 1:4] * 1000
    outputdata[:, 6] = predict_north
    outputdata[:, 7] = predict_east
    outputdata[:, 8] = predict_up
    outputdata[:, 9] = filter_north
    outputdata[:, 10] = filter_east
    outputdata[:, 11] = filter_up
    data4origin = getpath('data4origin')
    print(str(std_r_north), str(std_r_east), str(std_r_up), str(std_p_north), str(std_p_east), str(std_p_up),
          str(std_f_north), str(std_f_east), str(std_f_up), str(std_r), str(std_p), str(std_f))
    np.savetxt(data4origin + "\\" + modelname[0:modelname.index('.')] + ".txt", outputdata, fmt="%f", delimiter=",")


def analysisrfralldays(feature, label, dirname):
    rfr_path = getpath("trained_path") + dirname + "\\"
    #
    daynums = np.arange(2, 51)
    outputdata = np.zeros((len(daynums), 5))
    i = 0
    for daynum in daynums:
        file = os.path.join(rfr_path, str(daynum) + '_rfr.pkl')
        print(str(i), ',', file)
        rfr = joblib.load(file)
        featshp = feature.shape
        predict = rfr.predict(feature.reshape((featshp[0], featshp[1] * featshp[2])))
        predict_north = predict[:, 0] * 1000
        predict_east = predict[:, 1] * 1000
        predict_up = predict[:, 2] * 1000
        refer_north = label[:, 1] * 1000
        refer_east = label[:, 2] * 1000
        refer_up = label[:, 3] * 1000
        filter_north = refer_north - predict_north
        filter_east = refer_east - predict_east
        filter_up = refer_up - predict_up
        std_f_north = np.std(filter_north)
        std_f_east = np.std(filter_east)
        std_f_up = np.std(filter_up)
        std_f = np.sqrt(std_f_north * std_f_north + std_f_east * std_f_east + std_f_up * std_f_up)
        outputdata[i, 0] = daynum - 1
        outputdata[i, 1] = std_f_north
        outputdata[i, 2] = std_f_east
        outputdata[i, 3] = std_f_up
        outputdata[i, 4] = std_f
        i = i + 1
    plt.plot(outputdata[:, 0], outputdata[:, 1], 'r', label='north')
    plt.plot(outputdata[:, 0], outputdata[:, 2], 'g', label='east')
    plt.plot(outputdata[:, 0], outputdata[:, 3], 'b', label='up')
    plt.plot(outputdata[:, 0], outputdata[:, 4], 'k', label='total')
    plt.legend()
    plt.ylabel('std(mm)')
    plt.xlabel('time(day)')
    plt.title('bjfs_rfr_analysisalldays')
    plt.show()
    data4origin = getpath('data4origin')
    np.savetxt(data4origin + "\\bjfs_" + dirname + "_analysisalldays.txt", outputdata, fmt="%f", delimiter=",")


def analysissvr_rbfalldays(feature, label):
    rfr_path = getpath("trained_path") + "svr_rbf\\"
    #
    daynums = np.arange(2, 51)
    outputdata = np.zeros((len(daynums), 5))
    i = 0
    for daynum in daynums:
        file = os.path.join(rfr_path, str(daynum) + '_svr_rbf.pkl')
        print(str(i), ',', file)
        rfr = joblib.load(file)
        featshp = feature.shape
        predict = rfr.predict(feature.reshape((featshp[0], featshp[1] * featshp[2])))
        predict_north = predict[:, 0] * 1000
        predict_east = predict[:, 1] * 1000
        predict_up = predict[:, 2] * 1000
        refer_north = label[:, 1] * 1000
        refer_east = label[:, 2] * 1000
        refer_up = label[:, 3] * 1000
        filter_north = refer_north - predict_north
        filter_east = refer_east - predict_east
        filter_up = refer_up - predict_up
        std_f_north = np.std(filter_north)
        std_f_east = np.std(filter_east)
        std_f_up = np.std(filter_up)
        std_f = np.sqrt(std_f_north * std_f_north + std_f_east * std_f_east + std_f_up * std_f_up)
        outputdata[i, 0] = daynum - 1
        outputdata[i, 1] = std_f_north
        outputdata[i, 2] = std_f_east
        outputdata[i, 3] = std_f_up
        outputdata[i, 4] = std_f
        i = i + 1
    plt.plot(outputdata[:, 0], outputdata[:, 1], 'r', label='north')
    plt.plot(outputdata[:, 0], outputdata[:, 2], 'g', label='east')
    plt.plot(outputdata[:, 0], outputdata[:, 3], 'b', label='up')
    plt.plot(outputdata[:, 0], outputdata[:, 4], 'k', label='total')
    plt.legend()
    plt.ylabel('std(mm)')
    plt.xlabel('time(day)')
    plt.title('bjfs_svr_rbf_analysisalldays')
    plt.show()
    data4origin = getpath('data4origin')
    np.savetxt(data4origin + "\\bjfs_svr_rbf_analysisalldays.txt", outputdata, fmt="%f", delimiter=",")


def analysiserfalldays(feature, label):
    erf_path = getpath("trained_path") + "erf\\"
    #
    daynums = np.arange(2, 51)
    outputdata = np.zeros((len(daynums), 5))
    i = 0
    for daynum in daynums:
        file = os.path.join(erf_path, str(daynum) + '_erf.pkl')
        print(str(i), ',', file)
        rfr = joblib.load(file)
        featshp = feature.shape
        predict = rfr.predict(feature.reshape((featshp[0], featshp[1] * featshp[2])))
        predict_north = predict[:, 0] * 1000
        predict_east = predict[:, 1] * 1000
        predict_up = predict[:, 2] * 1000
        refer_north = label[:, 1] * 1000
        refer_east = label[:, 2] * 1000
        refer_up = label[:, 3] * 1000
        filter_north = refer_north - predict_north
        filter_east = refer_east - predict_east
        filter_up = refer_up - predict_up
        std_f_north = np.std(filter_north)
        std_f_east = np.std(filter_east)
        std_f_up = np.std(filter_up)
        std_f = np.sqrt(std_f_north * std_f_north + std_f_east * std_f_east + std_f_up * std_f_up)
        outputdata[i, 0] = daynum - 1
        outputdata[i, 1] = std_f_north
        outputdata[i, 2] = std_f_east
        outputdata[i, 3] = std_f_up
        outputdata[i, 4] = std_f
        i = i + 1
    plt.plot(outputdata[:, 0], outputdata[:, 1], 'r', label='north')
    plt.plot(outputdata[:, 0], outputdata[:, 2], 'g', label='east')
    plt.plot(outputdata[:, 0], outputdata[:, 3], 'b', label='up')
    plt.plot(outputdata[:, 0], outputdata[:, 4], 'k', label='total')
    plt.legend()
    plt.ylabel('std(mm)')
    plt.xlabel('time(day)')
    plt.title('bjfs_erf_analysisalldays')
    plt.show()
    data4origin = getpath('data4origin')
    np.savetxt(data4origin + "\\bjfs_erf_analysisalldays.txt", outputdata, fmt="%f", delimiter=",")


def analysiscnn_multialldays(feature, label):
    erf_path = getpath("trained_path") + "cnn\\"
    #
    daynums = np.arange(2, 51)
    outputdata = np.zeros((len(daynums), 5))
    i = 0
    for daynum in daynums:
        file = os.path.join(erf_path, str(daynum) + '_day_cnnmulti.h5')
        print(str(i), ',', file)
        cnn = load_model(file)
        predict = cnn.predict(feature.reshape(feature.shape[0], 14, 32, 1))
        predict_north = predict[0].flatten() * 1000
        predict_east = predict[1].flatten() * 1000
        predict_up = predict[2].flatten() * 1000
        refer_north = label[:, 1] * 1000
        refer_east = label[:, 2] * 1000
        refer_up = label[:, 3] * 1000
        filter_north = refer_north - predict_north
        filter_east = refer_east - predict_east
        filter_up = refer_up - predict_up
        std_f_north = np.std(filter_north)
        std_f_east = np.std(filter_east)
        std_f_up = np.std(filter_up)
        std_f = np.sqrt(std_f_north * std_f_north + std_f_east * std_f_east + std_f_up * std_f_up)
        outputdata[i, 0] = daynum - 1
        outputdata[i, 1] = std_f_north
        outputdata[i, 2] = std_f_east
        outputdata[i, 3] = std_f_up
        outputdata[i, 4] = std_f
        i = i + 1
    plt.plot(outputdata[:, 0], outputdata[:, 1], 'r', label='north')
    plt.plot(outputdata[:, 0], outputdata[:, 2], 'g', label='east')
    plt.plot(outputdata[:, 0], outputdata[:, 3], 'b', label='up')
    plt.plot(outputdata[:, 0], outputdata[:, 4], 'k', label='total')
    plt.legend()
    plt.ylabel('std(mm)')
    plt.xlabel('time(day)')
    plt.title('bjfs_cnn_multi_analysisalldays')
    plt.show()
    data4origin = getpath('data4origin')
    np.savetxt(data4origin + "\\bjfs_cnn_multi_analysisalldays.txt", outputdata, fmt="%f", delimiter=",")
def viewPreproc(preprocpath,view_range):
    files = []
    wfiles = os.listdir(preprocpath)
    for file in wfiles:
        files.append(os.path.join(preprocpath, file))
    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # ax1, ax2, ax3 = axes.flat
    tab10 = mcolors.TABLEAU_COLORS
    tabkeys = list(tab10.keys())
    for i in view_range:
        print(files[i])
        data = np.load(files[i])
        label = data["label"]
        t = (label[:, 0]-label[0, 0])*7*24
        offset = (i-view_range[0])*8
        plt.plot(t, label[:, 1]*1000-offset, c=tab10[tabkeys[i%10]], label='north')
        plt.title(wfiles[i])
        # ax2.plot(label[:, 0], label[:, 2], 'g', label='east')
        # ax3.plot(label[:, 0], label[:, 3], 'b', label='up')
        datamat = np.zeros((len(t),4))
        datamat[:,0] = t
        datamat[:, 1] = label[:, 1]*1000
        datamat[:, 2] = label[:, 2] * 1000
        datamat[:, 3] = label[:, 3] * 1000
        txtfile = str(i)+'.txt'
        np.savetxt(txtfile, datamat)
    plt.show()
def viewnormdist_daynum(pred_res, predictname,cnnviewpath):
    # plt.figure(figsize=(64, 48), dpi=100)

    fig, axs = plt.subplots(3, 1, sharex=True)

    # Remove horizontal space between axes
    fsz =14
    fig.subplots_adjust(hspace=0)
    axs[0].tick_params(axis='y', labelsize=fsz-1 )
    axs[1].tick_params(axis='y', labelsize=fsz-1 )
    axs[2].tick_params(axis='y', labelsize=fsz-1 )
    plt.xticks(fontsize=fsz-1)
    axs[0].plot(pred_res[:,0], pred_res[:,1], '.-')
    axs[0].grid(True)
    axs[0].set_ylabel('North (mm)',fontsize=fsz)
    axs[1].plot(pred_res[:,0], pred_res[:,2], '.-')
    axs[1].grid(True)
    axs[1].set_ylabel('East (mm)',fontsize=fsz)
    axs[2].plot(pred_res[:,0], pred_res[:,3], '.-')
    axs[2].set_xticks(np.arange(0, 18, 3))
    axs[2].grid(True)
    axs[2].set_ylabel('Up (mm)',fontsize=fsz)
    axs[2].set_xlabel('Time (days)',fontsize=fsz)
    plt.savefig(cnnviewpath + predictname + '_1' + '.png', bbox_inches='tight', pad_inches=0.01)
    plt.show()

    fig, axs = plt.subplots(3, 1, sharex=True)
    fsz =14
    fig.subplots_adjust(hspace=0)
    axs[0].tick_params(axis='y', labelsize=fsz-1 )
    axs[1].tick_params(axis='y', labelsize=fsz-1 )
    axs[2].tick_params(axis='y', labelsize=fsz-1 )
    plt.xticks(fontsize=fsz-1)
    # Remove horizontal space between axes
    axs[0].plot(pred_res[:,0], pred_res[:,4], '.-')
    axs[0].grid(True)
    axs[0].set_ylabel('North (mm)',fontsize=fsz)
    axs[1].plot(pred_res[:,0], pred_res[:,5], '.-')
    axs[1].set_ylim(15,23)
    axs[1].grid(True)
    axs[1].set_ylabel('East (mm)',fontsize=fsz)
    axs[2].plot(pred_res[:,0], pred_res[:,6], '.-')
    axs[2].set_xticks(np.arange(0, 18, 2))
    axs[2].grid(True)
    axs[2].set_ylabel('Up (mm)',fontsize=fsz)
    axs[2].set_xlabel('Time (days)',fontsize=fsz)
    plt.savefig(cnnviewpath + predictname + '_2' + '.png', bbox_inches='tight', pad_inches=0.01)
    plt.show()
def  view_all_sts_compare(first_pred_res,second_pred_res,sts):
    for i in range(0, len(sts)):
        sts[i] = sts[i].upper()
    x = np.arange(len(sts))
    width = 0.35

    fig, axs = plt.subplots(1, 3, sharey=True)
    fsz=15
    axs[0].tick_params(axis='x', labelsize=fsz-1 )
    axs[0].tick_params(axis='y', labelsize=fsz-1 )
    axs[1].tick_params(axis='x', labelsize=fsz-1 )
    axs[2].tick_params(axis='x', labelsize=fsz-1 )
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()
    axs[2].invert_yaxis()
    # Remove horizontal space between axes
    fig.subplots_adjust(wspace=0)
    axs[0].barh(x - width / 2, first_pred_res[:,0], width, label='8 days',tick_label=sts)
    axs[0].barh(x + width / 2, first_pred_res[:,3], width, label='18 days')
    axs[0].set_ylabel('Station',fontsize=fsz)
    axs[0].set_xlabel('North (mm)',fontsize=fsz)
    axs[0].set_xlim(left=0, right=59)
    axs[0].set_xticks([0,25,50])
    # axs[0].set_yticks(x, sts.all)
    fig.legend( ncol = 2,loc='upper center',bbox_to_anchor=(0.5, 0.95),fontsize=fsz-1)
    axs[1].barh(x - width / 2, first_pred_res[:,1], width, label='8 days')
    axs[1].barh(x + width / 2, first_pred_res[:,4], width, label='18 days')
    axs[1].set_xlabel('East (mm)',fontsize=fsz)
    axs[1].set_xlim(left = 0, right = 39)
    axs[1].set_xticks([0,15,30])
    axs[2].barh(x - width / 2, first_pred_res[:,2], width, label='8 days')
    axs[2].barh(x + width / 2, first_pred_res[:,5], width, label='18 days')
    axs[2].set_xlim(left = 0, right = 169)
    axs[2].set_xlabel('Up (mm)',fontsize=fsz)
    axs[2].set_xticks([0,80,160])

    # plt.tight_layout()
    plt.savefig('all_sts_compare_01'+'.png',bbox_inches='tight',pad_inches=0.02)
    plt.show()

    fig, axs = plt.subplots(1, 3, sharey=True)
    axs[0].tick_params(axis='x', labelsize=fsz-1 )
    axs[0].tick_params(axis='y', labelsize=fsz-1 )
    axs[1].tick_params(axis='x', labelsize=fsz-1 )
    axs[2].tick_params(axis='x', labelsize=fsz-1 )
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()
    axs[2].invert_yaxis()
    # Remove horizontal space between axes
    fig.subplots_adjust(wspace=0)
    axs[0].barh(x - width / 2, second_pred_res[:,0], width, label='8 days',tick_label=sts)
    axs[0].barh(x + width / 2, second_pred_res[:,3], width, label='18 days')
    axs[0].set_xlim(left=0, right=59)
    axs[0].set_xticks([0,25,50])
    axs[0].set_ylabel('Station',fontsize=fsz)
    axs[0].set_xlabel('North (mm)',fontsize=fsz)
    fig.legend( ncol = 2,loc='upper center',bbox_to_anchor=(0.5, 0.95),fontsize=fsz)
    axs[1].barh(x - width / 2, second_pred_res[:,1], width, label='8 days')
    axs[1].barh(x + width / 2, second_pred_res[:,4], width, label='18 days')
    axs[1].set_xlim(left = 0, right = 35)
    axs[1].set_xticks([0,15,30])
    axs[1].set_xlabel('East (mm)',fontsize=fsz)
    # axs[1].set_ylim(bottom = 0, top = 7.9)
    axs[2].barh(x - width / 2, second_pred_res[:,2], width, label='8 days')
    axs[2].barh(x + width / 2, second_pred_res[:,5], width, label='18 days')
    axs[2].set_xlim(left = 0, right = 139)
    axs[2].set_xlabel('Up (mm)',fontsize=fsz)
    axs[2].set_xticks([0,60,120])
    plt.savefig('all_sts_compare_02'+'.png',bbox_inches='tight',pad_inches=0.02)
    plt.show()