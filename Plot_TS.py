# Program to plot time series

# Libraries

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import pandas as pd

# RMSE function
def rms_error(x, y):

    d = x - y
    d2 = d ** 2
    rmse_val = np.sqrt(d2.mean())
    return rmse_val

# Setting
dateformat = '%d.%m.%Y'

# Load data
data = np.load('./data/cnn_data.npz', allow_pickle=True)
data2=np.load('./data/meas_data.npz', allow_pickle=True)

# Get data
data = dict(data)
Headers = list(data.keys())
for i in range(0, len(Headers)):
    H = Headers[i]
    str2 = H + ' = data[' + '"' + H + '"' + ']'
    exec(str2)
    # print('Get info: ' + H)

Date = data2['Date']
depth = data2['depth']

# Function for data splitting
def data_split(df, pr_train, pr_test):
    pr_train = pr_train/100
    pr_test2 = (100 - pr_test) /100
    train, validate, test = np.split(df.sample(frac=1,random_state=0), [int(pr_train*len(df)), int(pr_test2*len(df))])
    return train, validate, test

# Function for preparing time for x axis
def mk_timeplot(x, n, dateformat):
    xtk=np.linspace(min(x),max(x),n)
    xtk2 = dt.num2date(xtk)
    xtk2 = pd.to_datetime(xtk2).strftime(dateformat)
    return xtk, xtk2

# Split time
x_train, x_val, x_test = data_split(pd.DataFrame(Date),70,15)
x_train = dt.date2num(x_train).ravel()
x_val = dt.date2num(x_val).ravel()
x_test = dt.date2num(x_test).ravel()

xtk_train, xtk2_train = mk_timeplot(x_train, 5, dateformat)
xtk_val, xtk2_val = mk_timeplot(x_val, 5, dateformat)
xtk_test, xtk2_test = mk_timeplot(x_test, 5, dateformat)

n1, n2 = np.shape(output_train)
rr = 0

# Plot data
font = {'family' : 'Times New Roman',
        #'weight' : 'bold',
        'size'   : 12}
plt.rc('font', **font)

fig, axs = plt.subplots(n2, 3, figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.2,bottom = 0.2)
axs = axs.ravel()
r=0

# Train
for i in range(0, n2):
    V1 = output_train[:, i]
    V2 = train_predictions[:, i]
    RMSE = rms_error(V1, V2)
    axs[rr].plot(x_train, V1, '.b')
    axs[rr].plot(x_train, V2, '.r')
    axs[rr].set_xlim([min(x_train), max(x_train)])
    axs[rr].set_xticks(xtk_train)

    if i == n2-1:
        axs[rr].set_xticklabels(xtk2_train, rotation=45)
    else:
        axs[rr].set_xticklabels([])
    if i == 0:
        axs[rr].set_title('Training \n RMSE= ' + '{:.3f}'.format(RMSE) + ' [%]')
    else:
        axs[rr].set_title('RMSE= ' + '{:.3f}'.format(RMSE) + ' [%]')


    axs[rr].set_ylabel('Depth=' + '{:.1f}'.format(depth[i]) + ' [m]' + '\n WC [%]')

    XLM = axs[rr].get_xlim()
    YLM = axs[rr].get_ylim()
    ss = chr(ord('a') + r)
    axs[rr].text(XLM[0], YLM[1], ' (' + ss + ')', verticalalignment='bottom')
    r = r + 1

    if i == n2 - 1:
        axs[rr].set_xlabel('Date')
    rr = rr + 1

# Validation
    V1 = output_val[:, i]
    V2 = val_predictions[:, i]
    RMSE = rms_error(V1, V2)
    axs[rr].plot(x_val, V1, '.b')
    axs[rr].plot(x_val, V2, '.r')
    axs[rr].set_xlim([min(x_val), max(x_val)])
    axs[rr].set_xticks(xtk_val)
    if i == n2-1:
        axs[rr].set_xticklabels(xtk2_val, rotation=45)
    else:
        axs[rr].set_xticklabels([])
    if i == 0:
        axs[rr].set_title('Validation \n RMSE= ' + '{:.3f}'.format(RMSE) + ' [%]')
    else:
        axs[rr].set_title('RMSE= ' + '{:.3f}'.format(RMSE) + ' [%]')

    XLM = axs[rr].get_xlim()
    YLM = axs[rr].get_ylim()
    ss = chr(ord('a') + r)
    axs[rr].text(XLM[0], YLM[1], ' (' + ss + ')', verticalalignment='bottom')
    r = r + 1

    if i == n2 - 1:
        axs[rr].set_xlabel('Date')
    rr = rr + 1

# Test
    V1 = output_test[:, i]
    V2 = test_predictions[:, i]
    RMSE = rms_error(V1, V2)
    axs[rr].plot(x_test, V1, '.b')
    axs[rr].plot(x_test, V2, '.r')
    axs[rr].set_xlim([min(x_test), max(x_test)])
    axs[rr].set_xticks(xtk_test)
    if i == n2 - 1:
        axs[rr].set_xticklabels(xtk2_test, rotation=45)
    else:
        axs[rr].set_xticklabels([])
    if i == 0:
        axs[rr].set_title('Test \n RMSE= ' + '{:.3f}'.format(RMSE) + ' [%]')
    else:
        axs[rr].set_title('RMSE= ' + '{:.3f}'.format(RMSE) + ' [%]')

    XLM = axs[rr].get_xlim()
    YLM = axs[rr].get_ylim()
    ss = chr(ord('a') + r)
    axs[rr].text(XLM[0], YLM[1], ' (' + ss + ')', verticalalignment='bottom')
    r = r + 1

    if i == n2 - 1:
        axs[rr].set_xlabel('Date')
    rr = rr + 1

plt.show()
