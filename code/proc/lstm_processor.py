import os
import pickle

import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import load_model
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizer_v2.adam import Adam
from matplotlib import pyplot as plt
from tensorflow.python.keras.preprocessing.timeseries import timeseries_dataset_from_array

from analyst.viewhandler import view_common_predict


# construct model
def lstm_model():
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(32, 13)))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mse', optimizer=Adam())
    model_lstm.summary()
    return model_lstm
def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def lstm_train(preprocpath, lstmworkpath, train_range):
    step = 6

    past = 2880
    future = 72
    learning_rate = 0.001
    batch_size = 256
    epochs = 10

    # lstmmodelpath = lstmworkpath + '\\model\\'
    # lstmhistpath = lstmworkpath + '\\hist\\'
    files = []
    wfiles = os.listdir(preprocpath)
    for file in wfiles:
        files.append(os.path.join(preprocpath, file))
    # files.reverse()
    feature = None
    label = None
    for i in train_range:
        print('train:', files[i])
        data = np.load(files[i])
        eachfeature = data["feature"]
        eachfeature[:, 12, :] = 0
        eachlabel = data["label"]
        eachfeature = eachfeature.reshape(-1, 13 * 32)
        if i == train_range[0]:
            feature = eachfeature
            label = eachlabel
        else:
            feature = np.append(feature, eachfeature, axis=0)
            label = np.append(label, eachlabel, axis=0)

    split_fraction = 0.715
    train_split = int(split_fraction * int(feature.shape[0]))
    train_data = feature[0: train_split - 1]
    val_data = feature[train_split:]
    start = past + future
    end = start + train_split
    train_label = label[start:end][:, 1:4]
    sequence_length = int(past / step)
    dataset_train = timeseries_dataset_from_array(
        train_data,
        train_label,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )
    x_end = len(val_data) - past - future

    label_start = train_split + past + future

    x_val = val_data[:x_end]
    y_val = label[label_start:][:, 1:4]

    dataset_val = timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    for batch in dataset_train.take(1):
        inputs, targets = batch
    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)
    inputs = Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = LSTM(32)(inputs)
    outputs = Dense(3)(lstm_out)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    model.summary()


    path_checkpoint = "model_checkpoint.h5"
    es_callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    modelckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback],
    )


    visualize_loss(history, "Training and Validation Loss")
    # feature = feature.reshape(feature.shape[0], 13, 32, 1)
    #
    #
    #
    # for j in range(1, 4):
    #     # 1,north
    #     # 2,east
    #     # 3,up
    #     modelname = str(train_range[0]) + '_' + str(train_range[-1]) + '_' + str(j)
    #     # if os.path.exists(cnnmodelpath + modelname + ".h5"):
    #     #     print('skip:'+cnnmodelpath + modelname + ".h5")
    #     #     continue
    #     model = lstm_model()
    #     early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
    #     history = model.fit(feature, label[:, j]*1000, epochs=300, batch_size=64, validation_split=0.125, verbose=2,
    #                         shuffle=True, callbacks=[early_stopping])
    #     # 保存训练的模型及历史记录
    #     with open(lstmhistpath + modelname + ".pkl", 'wb') as hist_file:
    #         pickle.dump(history.history, hist_file)
    #     model.save(lstmmodelpath + modelname + ".h5")
    #     view_cnn_train_hist(lstmworkpath, modelname)
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
