import os
import pickle

# import joblib
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
# from dtw import *
# import scipy.io as sio
# from keras.engine.saving import load_model
# from scipy import signal
# from wavelets import WaveletAnalysis

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
    # plt.savefig(getpath('model_view_temp') + file + '_ned.png')
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