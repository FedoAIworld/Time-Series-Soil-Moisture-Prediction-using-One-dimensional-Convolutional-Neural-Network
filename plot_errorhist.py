# Function to plot histogram of the residual errors

# Library
import matplotlib.pyplot as plt

def plot_errorhist(nbins, M1_train, M2_train, M1_val, M2_val):

    error_train = M2_train.ravel() - M1_train.ravel()
    error_val = M2_val.ravel() - M1_val.ravel()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.hist(error_train, bins = nbins, color='b')
    ax1.set_title('Training')
    ax1.set_ylabel('Frequency')

    ax2.hist(error_val, bins = nbins, color='b')
    ax2.set_title('Validation')
    ax2.set_ylabel('Frequency')

    plt.show()
