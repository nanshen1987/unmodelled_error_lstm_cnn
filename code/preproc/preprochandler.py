import os

import numpy as np

# 获取特征数据归一化参数
from config.configutil import getpath
from util.const import const
from util.coordinateutils import CoordinateTransform
import numpy as np
import matplotlib.pyplot as plt

from util.mathutils import get_rms


def getsatregularparms(satInfo):
    satitemnum = satInfo.shape[0]
    starttime = satInfo['week'][0] + satInfo['tow'][0] / (7 * 24 * 3600)
    endtime = satInfo['week'][satitemnum - 1] + satInfo['tow'][satitemnum - 1] / (7 * 24 * 3600)
    featuremeans = np.array(
        [starttime, 0, 0, 0, satInfo['resp'].mean(), satInfo['resc'].mean(), 0, satInfo['snr'].mean(), 0, 0,
         0, 0, 0, 0])
    featurestds = [endtime - starttime, 1, 360, 180, satInfo['resp'].std(), satInfo['resc'].std(), 1,
                   satInfo['snr'].mean(), 1, 1,
                   1 if satInfo['lock'].max() < 1 else satInfo['lock'].max(),
                   1 if satInfo['outc'].max() < 1 else satInfo['outc'].max(),
                   1 if satInfo['slipc'].max() < 1 else satInfo['slipc'].max(),
                   1 if satInfo['rejc'].max() < 1 else satInfo['rejc'].max()]
    return featuremeans, featurestds


def getGlobalNormalizedParams(satInfo):
    # min max mean std
    normalizedParams = np.zeros([13, 4])
    satitemnum = satInfo.shape[0]
    # Gps time
    mintime = satInfo['week'][0] + satInfo['tow'][0] / (7 * 24 * 3600)
    maxtime = satInfo['week'][satitemnum - 1] + satInfo['tow'][satitemnum - 1] / (7 * 24 * 3600)
    normalizedParams[0, 0] = mintime
    normalizedParams[0, 1] = maxtime
    # azimuth
    normalizedParams[1, 0] = satInfo['az'].min()
    normalizedParams[1, 1] = satInfo['az'].max()
    normalizedParams[1, 2] = satInfo['az'].mean()
    normalizedParams[1, 3] = satInfo['az'].std()
    # elevation
    normalizedParams[2, 0] = satInfo['el'].min()
    normalizedParams[2, 1] = satInfo['el'].max()
    normalizedParams[2, 2] = satInfo['el'].mean()
    normalizedParams[2, 3] = satInfo['el'].std()
    # resp
    normalizedParams[3, 0] = satInfo['resp'].min()
    normalizedParams[3, 1] = satInfo['resp'].max()
    normalizedParams[3, 2] = satInfo['resp'].mean()
    normalizedParams[3, 3] = satInfo['resp'].std()
    # resc
    normalizedParams[4, 0] = satInfo['resc'].min()
    normalizedParams[4, 1] = satInfo['resc'].max()
    normalizedParams[4, 2] = satInfo['resc'].mean()
    normalizedParams[4, 3] = satInfo['resc'].std()
    # snr
    normalizedParams[5, 0] = satInfo['snr'].min()
    normalizedParams[5, 1] = satInfo['snr'].max()
    normalizedParams[5, 2] = satInfo['snr'].mean()
    normalizedParams[5, 3] = satInfo['snr'].std()
    # flag
    normalizedParams[6, 1] = 1
    normalizedParams[7, 0] = 1
    normalizedParams[7, 1] = 2
    normalizedParams[8, 1] = 3
    # count
    normalizedParams[9, 1] = 1 if satInfo['lock'].max() < 1 else satInfo['lock'].max()
    normalizedParams[10, 1] = 1 if satInfo['outc'].max() < 1 else satInfo['outc'].max()
    normalizedParams[11, 1] = 1 if satInfo['slipc'].max() < 1 else satInfo['slipc'].max()
    normalizedParams[12, 1] = 1 if satInfo['rejc'].max() < 1 else satInfo['rejc'].max()
    return normalizedParams


# 获取标签归一化参数
def getposregularparams(posInfo):
    positemnum = posInfo.shape[0]
    starttime = posInfo['week'][0] + posInfo['tow'][0] / (7 * 24 * 3600)
    endtime = posInfo['week'][positemnum - 1] + posInfo['tow'][positemnum - 1] / (7 * 24 * 3600)
    labelmeans = [starttime, posInfo['posx'].mean(), posInfo['posy'].mean(), posInfo['posz'].mean()]
    labelstds = [endtime - starttime, posInfo['posx'].std(), posInfo['posy'].std(), posInfo['posz'].std()]
    return labelmeans, labelstds


def getsolregularparams(solInfo):
    tf = CoordinateTransform(const.WGS_84_SEMI_A, const.WGS_84_SEMI_B)
    # xyzss = solInfo.ix[:, ['x', 'y', 'z']]
    xyzss = solInfo.loc[:, ['x', 'y', 'z']]
    mxyz = xyzss.mean().values
    mblh = tf.xyz2blh(mxyz)
    return (mxyz, mblh)


def preprocfeature(satInfo, posInfo, featuremeans, featurestds, labelmeans, labelstds):
    satitemnum = satInfo.shape[0]
    starttime = satInfo['week'][0] + satInfo['tow'][0] / (7 * 24 * 3600)
    # endtime = satInfo['week'][satitemnum - 1] + satInfo['tow'][satitemnum - 1] / (7 * 24 * 3600)
    curweek = satInfo['week'][0]
    curtow = satInfo['tow'][0]
    curtime = starttime
    positemnum = posInfo.shape[0]
    feature = np.zeros([positemnum, 14, 32])
    label = np.zeros([positemnum, 4])
    featureunit = np.zeros([14, 32])
    featureIdx = 0
    # 按照时间构造训练特征
    for i in range(satitemnum):
        dataline = satInfo.iloc[i]
        week = dataline['week']
        tow = dataline['tow']
        if week != curweek or tow != curtow:
            curweek = week
            curtow = tow
            curtime = week + tow / (7 * 24 * 3600)
            feature[featureIdx, :, :] = featureunit
            featureIdx += 1
            featureunit = np.zeros([14, 32])
        prn = int(dataline['sat'][1:])
        featureunit[0, prn - 1] = curtime
        featureunit[1:, prn - 1] = dataline[3:]
        featureunit[:, prn - 1] = (featureunit[:, prn - 1] - featuremeans) / featurestds
    feature[featureIdx, :, :] = featureunit
    # 按照时间构造标签
    for i in range(positemnum):
        dataline = posInfo.iloc[i]
        week = dataline['week']
        tow = dataline['tow']
        label[i, 0] = week + tow / (7 * 24 * 3600)
        label[i, 1:4] = dataline[3:6]
        label[i, :] = (label[i, :] - labelmeans) / labelstds
    return feature, label


def preProcFeatureLable(satInfo, solInfo, normalizedParams, mxyz, mblh):
    satitemnum = satInfo.shape[0]
    samplenum = solInfo.shape[0]
    feature = np.zeros([samplenum, 13, 32])
    label = np.zeros([samplenum, 4])
    featureIdx = 0
    # coordinate transform
    tf = CoordinateTransform(const.WGS_84_SEMI_A, const.WGS_84_SEMI_B)
    # 按照时间构造标签
    for i in range(samplenum):
        soldataline = solInfo.iloc[i]
        solweek = soldataline['week']
        soltow = soldataline['tow']
        labeltime = solweek + soltow / (7 * 24 * 3600)
        label[i, 0] = labeltime
        xyz = soldataline[2:5].values
        dvec = xyz - mxyz
        ned = tf.xyz2ned(dvec, mblh)
        label[i, 1:4] = ned
        for j in range(featureIdx, satitemnum):
            satdataline = satInfo.iloc[j]
            satweek = satdataline['week']
            sattow = satdataline['tow']
            sattime = satweek + sattow / (7 * 24 * 3600)
            prn = int(satdataline['sat'][1:])
            max_min=normalizedParams[:,1]-normalizedParams[:,0]
            if abs(sattime - labeltime) <= 0.5 / (7.0 * 24.0 * 3600.0):
                feature[i, 0, prn - 1] = (sattime - normalizedParams[0, 0]) / max_min[0]
                az = satdataline['az']
                feature[i, 1, prn - 1] = (az - normalizedParams[1, 0]) / max_min[1]
                el = satdataline['el']
                feature[i, 2, prn - 1] = (el - normalizedParams[2, 0]) / max_min[2]
                resp = satdataline['resp']
                feature[i, 3, prn - 1] = (resp - normalizedParams[3, 2]) / normalizedParams[3, 3]
                resc = satdataline['resc']
                feature[i, 4, prn - 1] = (resc - normalizedParams[4, 2]) / normalizedParams[4, 3]
                snr = satdataline['snr']
                feature[i, 5, prn - 1] = (snr - normalizedParams[5, 2]) / normalizedParams[5, 3]
                vsat = satdataline['vsat']
                feature[i, 6, prn - 1] = (vsat - normalizedParams[6, 0]) / max_min[6]
                slip = satdataline['slip']
                feature[i, 7, prn - 1] = (slip - normalizedParams[7, 0]) / max_min[7]
                fix = satdataline['fix']
                feature[i, 8, prn - 1] = (fix - normalizedParams[8, 0]) / max_min[8]
                lock = satdataline['lock']
                feature[i, 9, prn - 1] = (lock - normalizedParams[9, 0]) / max_min[9]
                outc = satdataline['outc']
                feature[i, 10, prn - 1] = (outc - normalizedParams[10, 0]) / max_min[10]
                slipc = satdataline['slipc']
                feature[i, 11, prn - 1] = (slipc - normalizedParams[11, 0]) / max_min[11]
                rejc = satdataline['rejc']
                feature[i, 12, prn - 1] = (rejc - normalizedParams[12, 0]) / max_min[12]
            elif sattime < labeltime:
                continue
            else:
                featureIdx = j
                break
        if np.all(feature[i, :, :] == 0):
            print("probe")
    return feature, label


def viewpreproced(file):
    preproced_path_each = getpath('preproced_path_each')
    data = np.load(preproced_path_each + file)
    label = data["label"]
    t = label[:, 0]
    x = label[:, 1] * 1000
    y = label[:, 2] * 1000
    z = label[:, 3] * 1000
    east = plt.plot(t, x, 'r', label='north,rms:' + str(get_rms(x)) + 'std:' + str(np.std(x)))
    north = plt.plot(t, y, 'g', label='east,rms:' + str(get_rms(y)) + 'std:' + str(np.std(y)))
    up = plt.plot(t, z, 'b', label='down,rms:' + str(get_rms(z)) + 'std:' + str(np.std(z)))
    plt.legend()
    plt.title('times series:' + file)
    plt.ylabel('displacement(mm)')
    plt.xlabel('time(week)')
    plt.show()
    return np.array([np.mean(x), np.mean(y), np.mean(z)])


def removeabnormal(dir, rmsthd):
    toremfiles = []
    for root, _, wfiles in os.walk(dir):
        for file in wfiles:
            data = np.load(os.path.join(root, file))
            label = data["label"]
            x = label[:, 1] * 1000
            y = label[:, 2] * 1000
            z = label[:, 3] * 1000
            if get_rms(x) > rmsthd or get_rms(y) > rmsthd or get_rms(z) > rmsthd:
                print('remove:' + file)
                toremfiles.append(os.path.join(root, file))
    for file in toremfiles:
        os.remove(file)
