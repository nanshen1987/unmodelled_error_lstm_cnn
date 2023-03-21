import os

# import joblib
import numpy as np
# loading data
from analyst.viewhandler import  view_common_predict


def common_predict(preprocpath, workpath, train_range, test_range, show):
    modelpath = workpath + '\\model\\'
    outpath = workpath + '\\out\\'
    files = []
    wfiles = os.listdir(preprocpath)
    for file in wfiles:
        files.append(os.path.join(preprocpath, file))
    # files.reverse()
    for i in test_range:
        # 对比预测值与测量值
        data = np.load(files[i])
        print('test:',files[i])
        test_feature = data["feature"]
        featshp = test_feature.shape
        test_feature = test_feature.reshape((featshp[0], featshp[1] * featshp[2]))
        for j in range(1, 4):
            # predict
            modelname = str(train_range[0]) + '_' + str(train_range[-1]) + '_' + str(j)
            predictname = str(train_range[0]) + '_' + str(train_range[-1]) + '_' + str(i) + '_' + str(j)
            rfr = joblib.load(modelpath + modelname + ".pkl")
            predict = rfr.predict(test_feature)
            predict = predict.flatten()
            refer = data["label"][:, j]
            filtered = refer - predict
            t = data["label"][:, 0]
            refer = refer * 1000
            predict = predict * 1000
            filtered = filtered * 1000
            # output
            np.savez(outpath + predictname, t=t, refer=refer, predict=predict, filtered=filtered)
            # show
            if show:
                view_common_predict(workpath, predictname)