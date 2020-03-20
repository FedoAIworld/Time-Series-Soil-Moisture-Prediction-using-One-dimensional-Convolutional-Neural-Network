# Program to evaluate CNN

# Libraries
import scipy.io as sio
import pandas as pd
import numpy as np
from tensorflow import keras
from plot_history import *
from print_correlations import *
from plot_errorhist import *
import pickle

# Load and get data
data = np.load('./data/cnn_data.npz', allow_pickle=True)
model = keras.models.load_model('./data/cnn_model.h5')
with open('./data/tr_history.pickle', 'rb') as f:
    tr_history = pickle.load(f)
data = dict(data)
Headers = list(data.keys())
for i in range(0, len(Headers)):
    H = Headers[i]
    str2 = H + ' = data[' + '"' + H + '"' + ']'
    exec(str2)
    #print('Get info: ' + H)

M1_train = np.array(output_train)
M1_val = np.array(output_val)
M1_test = np.array(output_test)
M2_train = np.array(train_predictions)
M2_val = np.array(val_predictions)
M2_test = np.array(test_predictions)

# Print layers
print(model.summary())
plot_history(tr_history, str_metrics, str_labels)
plot_errorhist(50, M1_train, M2_train, M1_val, M2_val)
print_correlations(M1_train, M2_train, M1_val, M2_val,M1_test, M2_test)
