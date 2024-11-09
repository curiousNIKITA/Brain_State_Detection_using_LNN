from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow.python
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical 
from util import load_model

# Load the dataset
data = pd.read_csv("emotions.csv")
scaler = StandardScaler()
label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
data['label'] = data['label'].replace(label_mapping)
X = data.drop('label', axis=1)
y = data['label'].copy()
scaler.fit(X)
X = scaler.transform(X) 
y = to_categorical(y)

# Reshaping the independent variables
X_resized = np.reshape(X, (X.shape[0],1,X.shape[1]))

# Loading the lnn model
model = load_model()
model.load_weights("test.h5")

app = Flask(__name__)
@app.route('/predict/<int:selected_row>', methods=['GET'])
def prediction_func(selected_row):
    row_data = X_resized[selected_row:selected_row+1, : ]
    predict =  model.predict(row_data)
    predict_classes = np.argmax(predict,axis=1)
     
    predicted_class = predict_classes[0]  # Assuming predict_classes is a 1D array
    predicted_label = [key for key, val in label_mapping.items() if val == predicted_class][0]
    
    return jsonify({'predicted_class': int(predicted_class), 'predicted_label': predicted_label})
"""
@app.route('/predict', methods=['POST'])
def handle_prediction():
    data = request.get_json()
    selected_index = data['selected_index']
    
    # Call your model function to get prediction
    prediction, message = prediction_func(selected_index)
    
    # Return prediction result as JSON
    return jsonify({'prediction': prediction, 'message': message})
"""

# @app.route("/")
# def main():
#     return "Enter '/' with a number in the dange of 0 to 200"


if __name__ == '__main__':
    app.run(host='localhost', port=5001)