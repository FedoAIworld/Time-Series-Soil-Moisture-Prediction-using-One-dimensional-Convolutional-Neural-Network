# Function to print correlations for output parameters

# Libraries
import numpy as np
from rms_error import *

def print_correlations(M1_train, M2_train, M1_val, M2_val, M1_test, M2_test):

    print('R: Correlation Coefficient')
    print('RMSE: %')
    print()

    TR_train = np.corrcoef(M1_train.ravel(), M2_train.ravel())[0, 1]
    TR_val = np.corrcoef(M1_val.ravel(), M2_val.ravel())[0, 1]
    TR_test = np.corrcoef(M1_test.ravel(), M2_test.ravel())[0, 1]
    TRMS_train = rms_error(M1_train.ravel(), M2_train.ravel())
    TRMS_val = rms_error(M1_val.ravel(), M2_val.ravel())
    TRMS_test = rms_error(M1_test.ravel(), M2_test.ravel())

    print('Total Correlations:')
    print("R (Train),  R (Validation),  R (Test)")
    print('{0:1.3f} \t {1:1.3f} \t {2:1.3f}'.format(TR_train, TR_val, TR_test))
    print()
    print("RMSE (Train),  RMSE (Validation),  RMSE (Test)")
    print('{0:1.3f} \t {1:1.3f} \t {2:1.3f}'.format(TRMS_train, TRMS_val, TRMS_test))
    print()
    print('Correlations for each parameter:')
    print("R (Train),  R (Validation),  R (Test),   RMSE (Train),  RMSE (Validation),  RMSE (Test)")

    for ii in range(0, M1_train.shape[1]):
        R_train = np.corrcoef(M1_train[:, ii], M2_train[:, ii])[0, 1]
        R_val = np.corrcoef(M1_val[:, ii], M2_val[:, ii])[0, 1]
        R_test = np.corrcoef(M1_test[:, ii], M2_test[:, ii])[0, 1]

        RMS_train = rms_error(M1_train[:, ii], M2_train[:, ii])
        RMS_val = rms_error(M1_val[:, ii], M2_val[:, ii])
        RMS_test = rms_error(M1_test[:, ii], M2_test[:, ii])

        print('{0:1.3f} \t {1:1.3f} \t {2:1.3f} \t {3:1.3f} \t {4:1.3f} \t {5:1.3f}'.format(R_train, R_val, R_test, RMS_train, RMS_val, RMS_test))
