from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Flatten, Dropout

def build_cnn(data_length = 100, n_channel = 370, last_dense = 2):
    model = models.Sequential()
    model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(data_length, n_channel), padding='same'))
    model.add(layers.Conv1D(32, 3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv1D(32, 3, activation='relu', padding='same'))
    model.add(layers.Conv1D(32, 3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(64, 3, activation='relu', padding='same'))
    model.add(layers.Conv1D(64, 3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(64, 3, activation='relu', padding='same'))
    model.add(layers.Conv1D(64, 3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.5))
    
    model.add(Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(last_dense, activation='softmax'))
    return model

