#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Written by Yuichiro Yada
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

################################################
#　A code for Fig. 3. This code uses an output .pkl file from "prediction_realdata.py".
################################################


import numpy as np

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
plt.rcParams['figure.subplot.bottom'] = 0.15
plt.rcParams['font.family'] = 'Arial'
import matplotlib.cm as cm

from sklearn.metrics import mean_squared_error
import dill

from models import predictive_Z


if __name__ == "__main__":
    plt.rcParams['figure.subplot.bottom'] = 0.15
    plt.rcParams['figure.subplot.left'] = 0.2
    testID_list = []
    Z_logprob_results_featureomit = []
    Z_logprob_results_featureadd = []
    original_datadir = './'
    final_state_pkl = original_datadir + 'session_after_mcmc4real_final.pkl'
    dill.load_session(final_state_pkl)
    print('loaded: '+final_state_pkl)

    #################################################
    # prediction by respectively omitted features.
    #################################################

    for set_ind in range(total_test_samp_num):
        set_state_pkl = original_datadir + 'session_after_mcmc4real_testset'+str(set_ind)+'.pkl'
        dill.load_session(set_state_pkl)
        test_ID = both_data_df['individualID'].iloc[test_index]
        testID_list.append(test_ID)
        test_data_df = data_df.query('individualID in @test_ID')
        test_abeta_df = abeta_df.query('individualID in @test_ID')

        test_X = test_data_df.iloc[:, 3:].values
        test_Z = test_abeta_df[{use_region_name}].values

        feature_num = test_X.shape[1]

        log_prob = 0.0
        Z_points = np.arange(0,1.21,0.01)
        X_testsamp_num = test_X.shape[0]
        Z_logprob = np.zeros([feature_num, X_testsamp_num, Z_points.shape[0]])

        for feature_ind in range(feature_num):
            print("------------------------------")
            print('feature_ind: ' + str(feature_ind))
            print("------------------------------")
            omit_test_X = test_X.copy()
            omit_test_X[:, feature_ind] = np.zeros(omit_test_X[:, feature_ind].shape)
            for testsamp_ind in range(0, X_testsamp_num):
                for Z_ind, new_Z in enumerate(Z_points):
                    log_prob = predictive_Z(new_X_obs=omit_test_X[testsamp_ind], new_Z_obs=new_Z, grad_Z_correction_factor=grad_Z_correction_factor, posterior_samples=posterior_samples)
                    Z_logprob[feature_ind, testsamp_ind, Z_ind] = log_prob
        Z_logprob_results_featureomit.append(Z_logprob)
#    dill.dump_session("session_posthoc_feature_omit_cumadd.pkl")

    Z_points = np.arange(0, 1.21, 0.01)
    mse_z_pred = np.zeros([feature_num])
    std_z_pred = np.zeros([feature_num])
    cum_cnt = 0
    for feature_ind in range(feature_num):
        predicted_Z = np.zeros([total_test_samp_num, ])
        true_test_Z = np.zeros([total_test_samp_num, ])
        for testset_ind in range(total_test_samp_num):
            argmax_ind = np.argmax(Z_logprob_results_featureomit[testset_ind][feature_ind], axis=1)
            predicted_Z[testset_ind] = Z_points[argmax_ind]
            testset_str = 'testset' + str(testset_ind)
            test_ID = testID_list[testset_ind]
            test_abeta_df = abeta_df.query('individualID in @test_ID')
            true_test_Z[testset_ind] = test_abeta_df.iloc[:, 3].values
        mse_z_pred[feature_ind] = mean_squared_error(true_test_Z, predicted_Z)
        std_z_pred[feature_ind] = np.std(true_test_Z-predicted_Z)

    feature_mse = np.zeros([feature_num,3])
    feature_mse[:,0] = range(feature_num)
    feature_mse[:, 1] = mse_z_pred
    feature_mse[:, 2] = std_z_pred
    sorted_feature_mse = feature_mse[np.argsort(feature_mse[:,1])[::-1]]

    fig_feature_mse = plt.figure(figsize=(6,5))
    plt.rcParams['font.size'] = '18'
    plt.plot(sorted_feature_mse[:,1], 'ko')
    plt.ylim(0, 0.18)
    plt.yticks([0.00, 0.05, 0.10, 0.15])
    plt.xlabel('Feature')
    plt.ylabel('MSE')
    plt.savefig("omited_feature_mse.eps")
    fig_feature_mse.show()

#    dill.dump_session("session_posthoc_feature_omit_cumadd.pkl")

    #################################################
    # prediction by cumulative added features.
    #################################################

    for set_ind in range(total_test_samp_num):
        set_state_pkl = original_datadir + 'session_after_mcmc4real_testset'+str(set_ind)+'.pkl'
        dill.load_session(set_state_pkl)
        test_ID = both_data_df['individualID'].iloc[test_index]
        testID_list.append(test_ID)
        test_data_df = data_df.query('individualID in @test_ID')
        test_abeta_df = abeta_df.query('individualID in @test_ID')

        test_X = test_data_df.iloc[:, 3:].values
        test_Z = test_abeta_df[{use_region_name}].values

        feature_num = test_X.shape[1]

        log_prob = 0.0
        Z_points = np.arange(0,1.21,0.01)
        X_testsamp_num = test_X.shape[0]
        Z_logprob = np.zeros([feature_num, X_testsamp_num, Z_points.shape[0]])
        add_test_X = np.zeros(test_X.shape)
        for feature_ind in range(feature_num):
            print("------------------------------")
            print('cumulative_feature_ind: ' + str(feature_ind))
            print("------------------------------")
            add_test_X[:, int(sorted_feature_mse[feature_ind,0])] = test_X[:, int(sorted_feature_mse[feature_ind,0])]
            for testsamp_ind in range(0, X_testsamp_num):
                for Z_ind, new_Z in enumerate(Z_points):
                    log_prob = predictive_Z(new_X_obs=add_test_X[testsamp_ind], new_Z_obs=new_Z, grad_Z_correction_factor=grad_Z_correction_factor, posterior_samples=posterior_samples)
                    Z_logprob[feature_ind, testsamp_ind, Z_ind] = log_prob
        Z_logprob_results_featureadd.append(Z_logprob)
    dill.dump_session("session_posthoc_feature_omit_cumadd.pkl")

    Z_points = np.arange(0, 1.21, 0.01)
    mse_z_pred = np.zeros([feature_num])
    std_z_pred = np.zeros([feature_num])
    cum_cnt = 0
    for feature_ind in range(feature_num):
        predicted_Z = np.zeros([total_test_samp_num, ])
        true_test_Z = np.zeros([total_test_samp_num, ])
        for testset_ind in range(total_test_samp_num):
            argmax_ind = np.argmax(Z_logprob_results_featureadd[testset_ind][feature_ind], axis=1)
            predicted_Z[testset_ind] = Z_points[argmax_ind]
            testset_str = 'testset' + str(testset_ind)
            test_ID = testID_list[testset_ind]
            test_abeta_df = abeta_df.query('individualID in @test_ID')
            true_test_Z[testset_ind] = test_abeta_df.iloc[:, 3].values
        mse_z_pred[feature_ind] = mean_squared_error(true_test_Z, predicted_Z)
        std_z_pred[feature_ind] = np.std(true_test_Z - predicted_Z)

    fig_cumfeature_mse = plt.figure(figsize=(6, 5))
    plt.rcParams['font.size'] = '18'
    plt.plot(np.arange(1,12),mse_z_pred, 'ko-')
    plt.ylim([0.0, 0.45])
    plt.xlabel('Cumulative numbers of features')
    plt.ylabel('MSE')
    plt.savefig("cumulative_added_feature_mse.eps")
    fig_cumfeature_mse.show()

    dill.dump_session("session_posthoc_feature_omit_cumadd.pkl")