#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Written by Yuichiro Yada
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

################################################
#　A code for Fig. 2f. Before running this code, please run "prediction_realdata.py"
# with and without gradient_term, and set the resulting amyloid beta predictions
# to variables named "original_predicted_Z" and "wograd_ predicted_Z" respectively.
# Then, save these data as a .pkl file.
################################################


import numpy as np

import dill

import matplotlib.pyplot as plt
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.subplot.bottom'] = 0.15
plt.rcParams['figure.subplot.left'] = 0.2

if __name__ == "__main__":
    dill.load_session('./for_plot_gradient_term_effect.pkl')

    original_error = np.abs((original_predicted_Z - true_test_Z)[true_test_Z<0.66])
    wograd_error = np.abs((wograd_predicted_Z - true_test_Z)[true_test_Z < 0.66])

    fig_original_vs_wograd = plt.figure(figsize=(5,5))
    plt.rcParams['font.size'] = '16'
    plt.scatter(original_error, wograd_error, color='k')
    plt.plot(np.array([-0.05,1.05]),np.array([-0.05,1.05]),'k--', linewidth=1)
    plt.xlim(-0.05, 0.7)
    plt.ylim(-0.05, 0.7)
    plt.xlabel('Error by the original model')
    plt.ylabel('Error by the model w/o gradient term')
    plt.savefig("original_vs_wograd.eps")
    fig_original_vs_wograd.show()


