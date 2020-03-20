# Function to plot training history of the CNN

# Library
import matplotlib.pyplot as plt

def plot_history(hist, str_metrics, str_labels):
    n = len(str_metrics)
    if n > 1:
        fig, ax = plt.subplots(n, 1)
        for ii in range(0, n):
            str_tr = str_metrics[ii]
            str_val = 'val_' + str_metrics[ii]
            str_y = str_labels[ii]

            ax[ii].plot(hist['epoch'], hist[str_tr], '-b', label='Training Error', linewidth=2)
            ax[ii].plot(hist['epoch'], hist[str_val], '-r', label='Validation Error', linewidth=2)
            ax[ii].grid()
            ax[ii].set_xlabel('Epoch')
            ax[ii].set_ylabel(str_y)
            ax[ii].legend()
    else:
        fig = plt.figure()
        ax = plt.gca()
        str_tr = str_metrics[0]
        str_val = 'val_' + str_metrics[0]
        str_y = str_labels[0]

        ax.plot(hist['epoch'], hist[str_tr], '-b', label='Training Error', linewidth=2)
        ax.plot(hist['epoch'], hist[str_val], '-r', label='Validation Error', linewidth=2)
        ax.grid()
        ax.set_xlabel('Epoch')
        ax.set_ylabel(str_y)
        ax.legend()

    plt.show()
