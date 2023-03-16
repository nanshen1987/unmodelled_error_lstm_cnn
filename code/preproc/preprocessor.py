import os
from time import time

import numpy as np
import pandas as pd

from preproc.preprochandler import getsolregularparams, getGlobalNormalizedParams, preProcFeatureLable


def preproc(parsedsolpath, parsedstatpath, preprocpath, soltype):
    satparsedext = "_sat.pkl"
    solparsedext = "_sol.pkl"
    satfrms = []
    solfrms = []
    filenames = []
    for root, dir, files in os.walk(parsedsolpath):
        for file in files:
            filename = file[0:file.find('_')]
            filenames.append(filename)
    # 移除非收敛部分
    skipnum = 0
    # 计算所有数据归一化参数
    newfiles = []
    for file in filenames:
        solfrm = pd.read_pickle(parsedsolpath + file + solparsedext).iloc[skipnum:]
        if soltype == 'kinematic' and (solfrm['x'].std() > 0.1 or solfrm['y'].std() > 0.1 or solfrm['z'].std() > 0.1):
            print('abnormal sol->' + file + ':', solfrm['x'].std(), ',', solfrm['y'].std(), ',', solfrm['z'].std())
        if soltype == 'static' and (solfrm['x'].std() > 0.01 or solfrm['y'].std() > 0.01 or solfrm['z'].std() > 0.01):
            print('abnormal sol->' + file + ':', solfrm['x'].std(), ',', solfrm['y'].std(), ',', solfrm['z'].std())
        solfrms.append(solfrm)
        satfrms.append(pd.read_pickle(parsedstatpath + file + satparsedext))
        newfiles.append(file)
    filenames = newfiles
    catsatfrm = pd.concat(satfrms, axis=0, ignore_index=True)
    catsolfrms = pd.concat(solfrms, axis=0, ignore_index=True)
    normalizedParams = getGlobalNormalizedParams(catsatfrm)
    (mxyz, mblh) = getsolregularparams(catsolfrms)

    #  每天数据分开预处理-》增加后续处理的灵活性
    t_tag_begin = time()
    for file in filenames:
        satfrm = pd.read_pickle(parsedstatpath + file + satparsedext)
        solfrm = pd.read_pickle(parsedsolpath + file + solparsedext).iloc[skipnum:]
        (feature, label) = preProcFeatureLable(satfrm, solfrm, normalizedParams, mxyz, mblh)
        np.savez(preprocpath + file, feature=feature, label=label, normalizedParams=normalizedParams,
                 mxyz=mxyz, mblh=mblh)
        print(file + "数据预处理: ", time() - t_tag_begin)
    print("数据预处理: ", time() - t_tag_begin)


def removenoiseprefile(preprocpath, soltype):
    wfiles = os.listdir(preprocpath)
    toremovedfiles = []
    for file in wfiles:
        prefile = os.path.join(preprocpath, file)
        with np.load(prefile) as data:
            label = data["label"]
            lx = label[:, 1]
            ly = label[:, 2]
            lz = label[:, 3]
            if len(lx) == 0 or len(ly) == 0 or len(lz) == 0:
                print('remove file->', prefile)
                toremovedfiles.append(prefile)
                continue
            if soltype == 'static' and (np.std(lx) > 0.003 or np.std(ly) > 0.003 or np.std(lz) > 0.003):
                print('remove file->', prefile, np.std(lx), np.std(ly), np.std(lz))
                toremovedfiles.append(prefile)
                continue
            if soltype == 'kinematic' and (np.std(lx) > 0.1 or np.std(ly) > 0.1 or np.std(lz) > 0.1):
                print('remove file->', prefile, np.std(lx), np.std(ly), np.std(lz))
                toremovedfiles.append(prefile)
                continue
    print(len(toremovedfiles))
    for rfile in toremovedfiles:
        os.remove(rfile)
def countprefile(preprocpath):
    wfiles = os.listdir(preprocpath)
    return len(wfiles)