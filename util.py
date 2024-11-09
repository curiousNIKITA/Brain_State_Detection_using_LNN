import keras
from ncps.wirings import AutoNCP
from ncps.tf import LTC
import tensorflow as tf
from ncps import wirings
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping, ModelCheckpoint


def load_model():        
    timesteps = 1
    # X_train = (1705, 2548)
    input_shape = (timesteps, (2548))
    wiring = wirings.AutoNCP(100, 15)
    rnn_cell = LTC(wiring)
    num_classes = 3
    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Conv1D(filters=82, kernel_size=3, activation='relu', padding='causal'),
            # keras.layers.RNN(rnn_cell, return_sequences=True),
            # LTC(wiring, return_sequences=True),
            # keras.layers.RNN(rnn_cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False),
            LTC(wiring, return_sequences=True),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model