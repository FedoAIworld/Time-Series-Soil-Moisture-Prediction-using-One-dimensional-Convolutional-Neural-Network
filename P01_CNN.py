# One-dimensional convolutional neural network

# Libraries
import numpy as np
import pandas as pd
from tensorflow import keras
from numpy.random import seed
from tensorflow import set_random_seed
import pickle
from input_prep import *
from plot_history import *
from plot_errorhist import *
from print_correlations import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

# Load data
data = np.load('./data/meas_data.npz', allow_pickle=True)

# Get data
data = dict(data)
Headers = list(data.keys())

for i in range(0, len(Headers)):
    H = Headers[i]
    str2 = H + ' = data[' + '"' + H + '"' + ']'
    exec(str2)
    #print('Get info: ' + H)

input = pd.DataFrame(np.append(EC, T, axis=1))
output = WC.copy()*100 # convert to percentage
output = pd.DataFrame(output)

# Function for data splitting
def data_split(df, pr_train, pr_test):
    pr_train = pr_train/100
    pr_test2 = (100 - pr_test) /100
    train, validate, test = np.split(df.sample(frac=1,random_state=0), [int(pr_train*len(df)), int(pr_test2*len(df))])
    return train, validate, test

# Split the data into training, validation and test subsets
input_train, input_val, input_test = data_split(input,70,15)
output_train, output_val, output_test = data_split(output,70,15)

# print(input.head())

# Data normalization
scaler_in = MinMaxScaler(feature_range=(0, 1))
scaler_out = MinMaxScaler(feature_range=(0, 1))
scaler_in = scaler_in.fit(input)
scaler_out = scaler_out.fit(output)
input_train_norm = scaler_in.transform(input_train)
input_val_norm = scaler_in.transform(input_val)
input_test_norm = scaler_in.transform(input_test)
output_train_norm = scaler_out.transform(output_train)
output_val_norm = scaler_out.transform(output_val)
output_test_norm = scaler_out.transform(output_test)

# Number of input and outputs and features
n_features = 1
n_input, n_output = input.shape[1], output.shape[1]

# Make input
input_train_norm = input_prep(input_train_norm, n_features)
input_val_norm = input_prep(input_val_norm, n_features)
input_test_norm = input_prep(input_test_norm, n_features)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

# Set seed
# For keras
seed(0)

# For tensorflow
set_random_seed(0)

# Define model
model = keras.Sequential()

LRU = 0.01

model.add(keras.layers.Conv1D(filters=8, kernel_size=2, strides=1, padding="same", input_shape=(n_input, n_features)))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
model.add(keras.layers.Conv1D(filters=16, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
model.add(keras.layers.Conv1D(filters=32, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))


model.add(keras.layers.Conv1D(filters=16, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.Dropout(0.02))
model.add(keras.layers.Conv1D(filters=8, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.Dropout(0.01))
model.add(keras.layers.Conv1D(filters=4, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))

model.add(keras.layers.Flatten())

# Print layers
print(model.summary())

# Training setting
opt = keras.optimizers.Adam(lr=0.001)
str_metrics = ['mean_squared_error', 'mean_absolute_error']
str_labels = ['MSE', 'MAE']
loss = 'mse'

# Train CNN
model.compile(optimizer=opt, loss=loss, metrics=str_metrics)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(input_train_norm, output_train_norm, epochs=1000,
                    validation_data = (input_val_norm, output_val_norm),
                    verbose=0, callbacks=[early_stop, PrintDot()])

# Training history
tr_history = pd.DataFrame(history.history)
tr_history['epoch'] = history.epoch
print(tr_history)

# Make predictions
train_predictions_norm = model.predict(input_train_norm)
val_predictions_norm = model.predict(input_val_norm)
test_predictions_norm = model.predict(input_test_norm)
train_predictions = scaler_out.inverse_transform(train_predictions_norm)
val_predictions = scaler_out.inverse_transform(val_predictions_norm)
test_predictions = scaler_out.inverse_transform(test_predictions_norm)

# Make output
cnn_data = {'input_train': input_train, 'input_test': input_test,'input_val': input_val,
             'output_train': output_train, 'output_test': output_test, 'output_val': output_val,
             'test_predictions':test_predictions,'val_predictions':val_predictions,'train_predictions':train_predictions,
              'str_metrics': str_metrics, 'str_labels': str_labels}

# Save data
joblib.dump(scaler_in, './data/scaler_in.pkl')
joblib.dump(scaler_out, './data/scaler_out.pkl')
np.savez_compressed('./data/cnn_data',**cnn_data)
model.save('./data/cnn_model.h5')
with open('./data/tr_history.pickle', 'wb') as f:
     pickle.dump(tr_history, f)

# Convert to numpy array
M1_train = np.array(output_train)
M1_val = np.array(output_val)
M1_test = np.array(output_test)
M2_train = np.array(train_predictions)
M2_val = np.array(val_predictions)
M2_test = np.array(test_predictions)

# Evaluations
plot_history(tr_history, str_metrics, str_labels)
plot_errorhist(50, M1_train, M2_train, M1_val, M2_val)
print_correlations(M1_train, M2_train, M1_val, M2_val,M1_test, M2_test)

print(n_input)