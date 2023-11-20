#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Written by Yuichiro Yada
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

################################################
#　A code for Fig. 5e-f. This code uses output .pkl files from "prediction_simdata.py".
################################################


import numpy as np
import seaborn as sns

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
import matplotlib.cm as cm

from sklearn.metrics import accuracy_score
from scipy import stats

if __name__ == "__main__":
    plt.figure(figsize=(6, 4))
    plt.rcParams['font.size'] = '16'

    ratio = np.array([(200 - 176) / 200, (200 - 160) / 200, (200 - 120) / 200, (200-100) / 200, (200 - 80) / 200])
    pos = np.array([1,2,3,4,5])

    unsupervised_num_array = np.array([176, 160, 120, 100, 80])
    mse = np.zeros(5)

    datapath = './'
    sample_num = 240
    abeta_only_observed_num = 40
    for ratio_ind, unsupervised_num in enumerate(unsupervised_num_array):
        dill.load_session(datapath + "session_after_mcmc4real_unsupervised_num_" + str(unsupervised_num) + ".pkl")
        total_test_samp_num = sample_num - abeta_only_observed_num - unsupervised_num
        predicted_Z = np.zeros([total_test_samp_num, ])
        true_test_Z = np.zeros([total_test_samp_num, ])
        true_test_t = np.zeros([total_test_samp_num, ])
        Z_points = np.arange(0, 1.21, 0.01)
        cum_cnt = 0
        for testset_ind, (_, test_index) in enumerate(kf.split(X[supervised_ind], input_label[supervised_ind])):
            dill.load_session(datapath + "session_after_mcmc4real_unsupervised_num_" + str(unsupervised_num) + "testset" + str(testset_ind) + ".pkl")
            testsamp_num = len(Z_logprob_results[testset_ind])
            argmax_ind = np.argmax(Z_logprob_results[testset_ind], axis=1)
            predicted_Z[cum_cnt:cum_cnt + testsamp_num] = Z_points[argmax_ind]
            true_test_Z[cum_cnt:cum_cnt + testsamp_num] = supervised_test_Z[:, 0]
            true_test_t[cum_cnt:cum_cnt + testsamp_num] = supervised_test_t
            cum_cnt += testsamp_num
            print(testset_ind)
            print(test_index)
        predicted_Z = predicted_Z[true_test_t != 0]
        true_test_Z = true_test_Z[true_test_t != 0]
        true_test_t = true_test_t[true_test_t != 0]
        mse[ratio_ind] = mean_squared_error(true_test_Z, predicted_Z)

    plt.plot(ratio,mse,'ko-')
    plt.ylim([0,0.015])
    plt.xlabel('Supervised sample ratio')
    plt.ylabel('MSE')
    plt.savefig("sim_supervized_ratio_Fig2D.png")
    plt.savefig("sim_supervized_ratio_Fig2D.eps")
    plt.show()

    unsupervised_num = 100
    datapath = './'
    dill.load_session(datapath+"session_after_mcmc4real_unsupervised_num_" + str(unsupervised_num) + ".pkl")
    total_test_samp_num = sample_num - abeta_only_observed_num - unsupervised_num
    sub_predicted_Z = np.zeros([total_test_samp_num, ])
    sub_true_test_Z = np.zeros([total_test_samp_num, ])
    sub_true_test_t = np.zeros([total_test_samp_num, ])
    sub_true_test_s = np.zeros([total_test_samp_num, ])

    Z_points = np.arange(0, 1.21, 0.01)
    sub_cum_cnt = 0
    for testset_ind, (_, test_index) in enumerate(kf.split(X[supervised_ind], input_label[supervised_ind])):
        dill.load_session(datapath+"session_after_mcmc4real_unsupervised_num_" + str(unsupervised_num) + "testset" + str(testset_ind) + ".pkl")
        testsamp_num = len(Z_logprob_results[testset_ind])
        argmax_ind = np.argmax(Z_logprob_results[testset_ind], axis=1)
        sub_predicted_Z[sub_cum_cnt:sub_cum_cnt + testsamp_num] = Z_points[argmax_ind]
        sub_true_test_Z[sub_cum_cnt:sub_cum_cnt + testsamp_num] = supervised_test_Z[:, 0]
        sub_true_test_t[sub_cum_cnt:sub_cum_cnt + testsamp_num] = supervised_test_t
        sub_true_test_s[sub_cum_cnt:sub_cum_cnt + testsamp_num] = supervised_test_s
        sub_cum_cnt += testsamp_num
        # testset_ind += 1
        print(testset_ind)
        print(test_index)
    mse_z_pred = mean_squared_error(sub_true_test_Z, sub_predicted_Z)

    plt.figure(figsize=(6, 4))
    plt.rcParams['font.size'] = '16'
    predicted_type = (sub_predicted_Z == 0)
    unique_time = np.unique(sub_true_test_t)
    accuracy_time = np.zeros(unique_time.shape)
    for time_ind, timing in enumerate(unique_time):
        accuracy_time[time_ind] = accuracy_score(predicted_type[sub_true_test_t==timing],sub_true_test_s[sub_true_test_t==timing])
    plt.plot(np.int8(unique_time), accuracy_time, 'ko-')
    plt.xlabel('Age [m.o.]')
    plt.ylabel('Accuracy')
    plt.xticks([4,6,8,10,12,14,16])
    plt.ylim(0,1.05)
    plt.savefig("sim_TypeAccuracy_time_Fig2E.png")
    plt.savefig("sim_TypeAccuracy_time_Fig2E.eps")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.rcParams['font.size'] = '16'
    Z_error = np.abs(sub_predicted_Z - sub_true_test_Z)
    sns.boxplot(np.int8(sub_true_test_t[sub_true_test_s == 0]), Z_error[sub_true_test_s == 0], color='C0', width=0.5, fliersize=0)
    sns.stripplot(np.int8(sub_true_test_t[sub_true_test_s == 0]), Z_error[sub_true_test_s == 0], color='k', size=4)

    plt.xlabel('Age [m.o.]')
    plt.ylabel('Absolute error')
    plt.savefig("sim_Zerror_time_Fig2F.png")
    plt.savefig("sim_Zerror_time_Fig2F.eps")
    plt.show()

    tmp_t = sub_true_test_t[sub_true_test_s == 0]
    tmp_Z = Z_error[sub_true_test_s == 0]
    control = tmp_Z[tmp_t==4]
    groups = [tmp_Z[tmp_t==6], tmp_Z[tmp_t==8], tmp_Z[tmp_t==10], tmp_Z[tmp_t==12], tmp_Z[tmp_t==14], tmp_Z[tmp_t==16]]
    p_values = []

    # Perform Mann-Whitney U-test for each treatment vs. control
    for group in groups:
        _, p = stats.mannwhitneyu(control, group, alternative='two-sided')  # Change the alternative if needed
        p_values.append(p)

    # Holm's correction
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.sort(p_values)
    holm_adjusted = sorted_p_values * (len(groups) - np.arange(len(groups)))
    holm_p = np.empty_like(holm_adjusted)
    holm_p[sorted_indices] = holm_adjusted

    print("Original p-values:", p_values)
    print("Holm-adjusted p-values:", holm_p)
