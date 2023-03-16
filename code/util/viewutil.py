from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

from config.configutil import getpath


def viewpkresult(t, refer, predict, title):
    rmse = np.sqrt(mean_squared_error(refer, predict))
    plref = plt.plot(t, refer, 'r')
    plpred = plt.plot(t, predict, 'g')
    plt.title(title + '  rmse:' + str(rmse))
    plt.xlabel("time(s)")
    plt.ylabel("displacement(mm)")
    plt.legend((plref[0], plpred[0]), ('refer', 'predict'))
    plt.savefig(getpath('model_view_temp')+title+'_pred.png')
    plt.show()


def viewdeeplearnHis(history, title):
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='test')
    plt.legend()
    plt.title(title)
    plt.savefig(getpath('model_view_temp')+title+'_his.png')
    plt.show()
