import os
import pickle

import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizer_v2.adam import Adam

from analyst.viewhandler import view_cnn_train_hist, view_common_predict


# construct model
def cnn_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(13, 32, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    # model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1, init="normal"))
    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model


def cnn_train(preprocpath, cnnworkpath, train_range):
    cnnmodelpath = cnnworkpath + '\\model\\'
    cnnhistpath = cnnworkpath + '\\hist\\'
    files = []
    wfiles = os.listdir(preprocpath)
    for file in wfiles:
        files.append(os.path.join(preprocpath, file))
    # files.reverse()
    feature = None
    label = None
    for i in train_range:
        print('train:',files[i])
        data = np.load(files[i])
        eachfeature = data["feature"]
        eachfeature[:, 12, :] = 0
        eachlabel = data["label"]
        if i == train_range[0]:
            feature = eachfeature
            label = eachlabel
        else:
            feature = np.append(feature, eachfeature, axis=0)
            label = np.append(label, eachlabel, axis=0)
    # r3
    # filter
    bx = np.abs(label[:, 1]) < 0.5
    by = np.abs(label[:, 2]) < 0.5
    bz = np.abs(label[:, 3]) < 0.5
    bt = bx & by & bz
    # 训练
    label = label[bt,:]
    feature = feature[bt,:,:]
    feature = feature.reshape(feature.shape[0], 13, 32, 1)
    for j in range(1, 4):
        # 1,north
        # 2,east
        # 3,up
        modelname = str(train_range[0]) + '_' + str(train_range[-1]) + '_' + str(j)
        # if os.path.exists(cnnmodelpath + modelname + ".h5"):
        #     print('skip:'+cnnmodelpath + modelname + ".h5")
        #     continue
        model = cnn_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
        history = model.fit(feature, label[:, j]*1000, epochs=300, batch_size=64, validation_split=0.125, verbose=2,
                            shuffle=True, callbacks=[early_stopping])
        # 保存训练的模型及历史记录
        with open(cnnhistpath + modelname + ".pkl", 'wb') as hist_file:
            pickle.dump(history.history, hist_file)
        model.save(cnnmodelpath + modelname + ".h5")
        view_cnn_train_hist(cnnworkpath, modelname)
def cnn_train_mask(preprocpath, cnnworkpath, train_range,maskidx):
    cnnmodelpath = cnnworkpath + 'model\\'
    cnnhistpath = cnnworkpath + 'hist\\'
    files = []
    wfiles = os.listdir(preprocpath)
    for file in wfiles:
        files.append(os.path.join(preprocpath, file))
    # files.reverse()
    feature = None
    label = None
    for i in train_range:
        print('train:',files[i])
        data = np.load(files[i])
        eachfeature = data["feature"]
        if maskidx>=0:
            eachfeature[:, maskidx, :] = 0
        eachlabel = data["label"]
        if i == train_range[0]:
            feature = eachfeature
            label = eachlabel
        else:
            feature = np.append(feature, eachfeature, axis=0)
            label = np.append(label, eachlabel, axis=0)
    # r3
    # filter
    bx = np.abs(label[:, 1]) < 0.5
    by = np.abs(label[:, 2]) < 0.5
    bz = np.abs(label[:, 3]) < 0.5
    bt = bx & by & bz
    # 训练
    label = label[bt,:]
    feature = feature[bt,:,:]
    feature = feature.reshape(feature.shape[0], 13, 32, 1)
    for j in range(1, 4):
        # 1,north
        # 2,east
        # 3,up
        modelname = str(train_range[0]) + '_' + str(train_range[-1]) + '_' + str(j)
        # if os.path.exists(cnnmodelpath + modelname + ".h5"):
        #     print('skip:'+cnnmodelpath + modelname + ".h5")
        #     continue
        model = cnn_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
        history = model.fit(feature, label[:, j]*1000, epochs=300, batch_size=64, validation_split=0.125, verbose=2,
                            shuffle=True, callbacks=[early_stopping])
        # 保存训练的模型及历史记录
        with open(cnnhistpath + modelname + ".pkl", 'wb') as hist_file:
            pickle.dump(history.history, hist_file)
        model.save(cnnmodelpath + modelname + ".h5")
        view_cnn_train_hist(cnnworkpath, modelname)

def cnn_predict(preprocpath, cnnworkpath, train_range, test_range, show):
    cnnmodelpath = cnnworkpath + 'model\\'
    cnnoutpath = cnnworkpath + 'out\\'
    # load model
    files = []
    wfiles = os.listdir(preprocpath)
    for file in wfiles:
        files.append(os.path.join(preprocpath, file))
    normdist = []
    for i in test_range:
        # 对比预测值与测量值
        # print('test:',files[i])
        data = np.load(files[i])
        test_label = data["label"]
        test_feature = data["feature"]
        # r3
        # filter
        bx = np.abs(test_label[:, 1]) < 0.5
        by = np.abs(test_label[:, 2]) < 0.5
        bz = np.abs(test_label[:, 3]) < 0.5
        bt = bx & by & bz
        test_label = test_label[bt,:]
        test_feature = test_feature[bt,:,:]
        # test_feature[:,11,:] = 0

        test_feature = test_feature.reshape(test_feature.shape[0], 13, 32, 1)
        for j in range(1, 4):
            # predict
            modelname = str(train_range[0]) + '_' + str(train_range[-1]) + '_' + str(j)
            predictname = str(train_range[0]) + '_' + str(train_range[-1]) + '_' + str(i) + '_' + str(j)
            # print(cnnmodelpath + modelname + ".h5")
            if(not os.path.exists(cnnmodelpath + modelname + ".h5")):
                continue
            model = load_model(cnnmodelpath + modelname + ".h5",None,False)
            predict = model.predict(test_feature)
            t = test_label[:, 0]
            refer = test_label[:, j]*1000
            predict = predict.flatten()
            filtered = refer - predict
            # alignment = dtw(predict, refer)
            # normdist.append(alignment.normalizedDistance)
            diffstd = np.std(predict-refer)/np.sqrt(2)
            normdist.append(diffstd)
            # output
            np.savez(cnnoutpath + predictname, t=t, refer=refer, predict=predict, filtered=filtered)
            # show
            if show:
                # view_cnn_train_hist(cnnworkpath, modelname)
                view_common_predict(cnnworkpath, predictname, j)
    return normdist
def cnn_predict_mask(preprocpath, cnnworkpath, train_range, test_range, maskidx):
    cnnmodelpath = cnnworkpath + 'model\\'
    cnnoutpath = cnnworkpath + 'out\\'
    # load model
    files = []
    wfiles = os.listdir(preprocpath)
    for file in wfiles:
        files.append(os.path.join(preprocpath, file))
    normdist = []
    for i in test_range:
        # 对比预测值与测量值
        # print('test:',files[i])
        data = np.load(files[i])
        test_label = data["label"]
        test_feature = data["feature"]
        # r3
        # filter
        bx = np.abs(test_label[:, 1]) < 0.5
        by = np.abs(test_label[:, 2]) < 0.5
        bz = np.abs(test_label[:, 3]) < 0.5
        bt = bx & by & bz
        test_label = test_label[bt,:]
        test_feature = test_feature[bt,:,:]
        if maskidx>= 0:
            test_feature[:,maskidx,:] = 0
        test_feature = test_feature.reshape(test_feature.shape[0], 13, 32, 1)
        for j in range(1, 4):
            # predict
            modelname = str(train_range[0]) + '_' + str(train_range[-1]) + '_' + str(j)
            predictname = str(train_range[0]) + '_' + str(train_range[-1]) + '_' + str(i) + '_' + str(j)
            # print(cnnmodelpath + modelname + ".h5")
            if(not os.path.exists(cnnmodelpath + modelname + ".h5")):
                continue
            model = load_model(cnnmodelpath + modelname + ".h5",None,False)
            predict = model.predict(test_feature)
            t = test_label[:, 0]
            refer = test_label[:, j]*1000
            predict = predict.flatten()
            filtered = refer - predict
            # alignment = dtw(predict, refer)
            # normdist.append(alignment.normalizedDistance)
            diffstd = np.std(predict-refer)/np.sqrt(2)
            normdist.append(diffstd)
            # output
            np.savez(cnnoutpath + predictname, t=t, refer=refer, predict=predict, filtered=filtered)
    return normdist