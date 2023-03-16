from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import plot_model

# construct model
from config.configutil import getpath

# model_view_path = getpath('model_view_path')


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


model = cnn_model()
plot_model(model, to_file= 'demo.png')
