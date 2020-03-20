# Scatter plot program

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.lines as mlines

# Load data
data = np.load('cnn_data.npz', allow_pickle=True)
model = keras.models.load_model('cnn_model.h5')

# Get data
data = dict(data)
Headers = list(data.keys())
for i in range(0, len(Headers)):
    H = Headers[i]
    str2 = H + ' = data[' + '"' + H + '"' + ']'
    exec(str2)
    # print('Get info: ' + H)

# Plot results
n1,n2 = np.shape(output_train)
rr = 0
r=0

fig, axs = plt.subplots(n2,3, figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.2,bottom = 0.2)
axs = axs.ravel()

for ii in range(0,n2):
    V1 = output_train[:, ii]
    V2 = train_predictions[:, ii]
    R = np.corrcoef(V1, V2)
    R = R[0,1]
    axs[rr].scatter(V1, V2)
    if ii == 0:
        axs[rr].set_title('Training \n R= ' + '{:.3f}'.format(R))
    else:
        axs[rr].set_title('R= ' + '{:.3f}'.format(R))
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = axs[rr].transAxes
    line.set_transform(transform)
    axs[rr].add_line(line)
    axs[rr].set_ylabel('Predictions')

    XLM = axs[rr].get_xlim()
    YLM = axs[rr].get_ylim()
    ss = chr(ord('a') + r)
    axs[rr].text(XLM[0], YLM[1], ' (' + ss + ')', verticalalignment='bottom')
    r = r + 1

    if ii == n2-1:
        axs[rr].set_xlabel('True Values')
    rr=rr+1

    V1 = output_val[:, ii]
    V2 = val_predictions[:, ii]
    R = np.corrcoef(V1, V2)
    R = R[0,1]
    axs[rr].scatter(V1, V2)
    if ii == 0:
        axs[rr].set_title('Validation \n R= ' + '{:.3f}'.format(R))
    else:
        axs[rr].set_title('R= ' + '{:.3f}'.format(R))
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = axs[rr].transAxes
    line.set_transform(transform)
    axs[rr].add_line(line)

    XLM = axs[rr].get_xlim()
    YLM = axs[rr].get_ylim()
    ss = chr(ord('a') + r)
    axs[rr].text(XLM[0], YLM[1], ' (' + ss + ')', verticalalignment='bottom')
    r = r + 1

    if ii == n2-1:
        axs[rr].set_xlabel('True Values')
    rr=rr+1

    V1 = output_test[:, ii]
    V2 =test_predictions[:, ii]
    R = np.corrcoef(V1, V2)
    R = R[0,1]
    axs[rr].scatter(V1, V2)
    if ii == 0:
        axs[rr].set_title('Test \n R= ' + '{:.3f}'.format(R))
    else:
        axs[rr].set_title('R= ' + '{:.3f}'.format(R))
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = axs[rr].transAxes
    line.set_transform(transform)
    axs[rr].add_line(line)

    XLM = axs[rr].get_xlim()
    YLM = axs[rr].get_ylim()
    ss = chr(ord('a') + r)
    axs[rr].text(XLM[0], YLM[1], ' (' + ss + ')', verticalalignment='bottom')
    r = r + 1

    if ii == n2-1:
        axs[rr].set_xlabel('True Values')
    rr=rr+1

plt.show()
