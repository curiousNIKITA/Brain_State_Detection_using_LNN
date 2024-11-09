import os 
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical 
import streamlit as st
import keras
from ncps.wirings import AutoNCP
from ncps.tf import LTC
import tensorflow as tf
from ncps import wirings
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping, ModelCheckpoint

"""## Summary
This code defines a function named `load_data` that takes an input file as a parameter. The function reads the data from the input file using pandas, performs data preprocessing using StandardScaler from scikit-learn, and prepares the data for training a neural network using to_categorical from Keras. The function returns the preprocessed input data, the corresponding labels, and a sample data point.

## Example Usage
```python
X, y, sample = load_data('input.csv')
```

## Code Analysis
### Inputs
- `input_file`: a string representing the path to the input file containing the data.
___
### Flow
1. Read the data from the input file using pandas.
2. Create an instance of StandardScaler to standardize the input data.
3. Extract a sample data point from the first row of the data.
4. Create a mapping of label names to numerical values.
5. Replace the label column in the data with the corresponding numerical values using the label mapping.
6. Separate the input features (X) from the labels (y).
7. Fit the scaler on the input features to compute the mean and standard deviation.
8. Transform the input features using the fitted scaler to standardize them.
9. Convert the labels to categorical format using one-hot encoding.
10. Return the preprocessed input features (X), labels (y), and the sample data point.
___
### Outputs
- `X`: a numpy array representing the preprocessed input features.
- `y`: a numpy array representing the labels in categorical format.
- `sample`: a pandas Series representing a sample data point from the input data.
___
"""


def load_data(input_file: str):
    """
    Load and preprocess data from an input file.
    
    Args:
        input_file (str): The path to the input file containing the data.
        
    Returns:
        X (numpy array): The preprocessed input features.
        y (numpy array): The labels in categorical format.
        sample (pandas Series): A sample data point from the input data.
    """
    # Read the data from the input file using pandas
    data = pd.read_csv(input_file)
    
    # Extract a sample data point from the first row of the data
    # sample = data.iloc[0]
    sample = data.loc[0, 'fft_0_b':'fft_749_b']
    
    # Create a mapping of label names to numerical values
    # label_mapping = {label: i for i, label in enumerate(data['label'].unique())}
    label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    

    # Replace the label column in the data with the corresponding numerical values using the label mapping
    # data['label'] = data['label'].map(label_mapping)
    data['label'] = data['label'].replace(label_mapping)
    # Separate the input features (X) from the labels (y)
    X = data.drop('label', axis=1).values
    y = data['label'].values
    
    # Create an instance of StandardScaler to standardize the input data
    scaler = StandardScaler()
    
    # Fit the scaler on the input features to compute the mean and standard deviation
    scaler.fit(X)
    
    # Transform the input features using the fitted scaler to standardize them
    X = scaler.transform(X)
    
    # Convert the labels to categorical format using one-hot encoding
    y = to_categorical(y)
    
    return X, y, sample


def load_model():
    wiring = wirings.AutoNCP(100, 15)
    num_classes = 3
    timesteps = 1
    input_shape = (timesteps, 2548)  # X_train.shape[2]
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
    # model.load_weights("test.h5")
    return model

def feeder():
    pass 