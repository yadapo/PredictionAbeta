#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Written by Yuichiro Yada
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

################################################
#　A code for Supplementary Figure 3
################################################


import numpy as np
import pandas as pd
import seaborn as sns
import os

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
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.optimize import curve_fit


from numpyro.infer import MCMC, NUTS, HMCGibbs, initialization
from scipy.stats import norm, truncnorm, gamma, mannwhitneyu
from jax import random

from functools import partial, reduce

import dill
import arviz as az

from models import logistic, latent_model, full_model, _gibbs_fn_full, predictive_Z

if __name__ == "__main__":
    # loading individual data
    individual_metadata_df = pd.read_csv("./ScientificData_5xFAD/5xFAD_individual_metadata.csv")
    individual_metadata_df = individual_metadata_df[{"individualID", "ageDeath", "individualCommonGenotype"}]
    individual_metadata_df = individual_metadata_df.rename(columns={'ageDeath': 'age', 'individualCommonGenotype': 'type'})
    individual_metadata_df["type"].loc[individual_metadata_df["type"] == "5XFAD"] = 0
    individual_metadata_df["type"].loc[individual_metadata_df["type"] == "C57BL6J"] = 1

    # loading mouse abeta data
    abeta_insoluble_df = pd.read_csv("./ScientificData_5xFAD/5xFAD_ab_levels insoluble_fraction.csv")
    abeta_insoluble_cortex_df = abeta_insoluble_df[abeta_insoluble_df["specimenID"].str.contains('cif')]
    abeta_insoluble_cortex_df = abeta_insoluble_cortex_df[{"individualID", "Abeta40", "Abeta42"}]
    abeta_insoluble_cortex_df["cortex"] = abeta_insoluble_cortex_df["Abeta42"]
    abeta_insoluble_cortex_df = abeta_insoluble_cortex_df.dropna()

    abeta_insoluble_hip_df = abeta_insoluble_df[abeta_insoluble_df["specimenID"].str.contains('hif')]
    abeta_insoluble_hip_df = abeta_insoluble_hip_df[{"individualID", "Abeta40", "Abeta42"}]
    abeta_insoluble_hip_df["hippocampus"] = abeta_insoluble_hip_df["Abeta42"]
    abeta_insoluble_hip_df = abeta_insoluble_hip_df.dropna()

    abeta_plaque_df = pd.read_csv("./ScientificData_5xFAD/5xFAD 4, 8 12 and 18 months_ ThioS-NeuN.csv")
    abeta_plaque_df = abeta_plaque_df[{"individualID","#objects/sqmm"}]
    abeta_plaque_df["plaque"] = abeta_plaque_df["#objects/sqmm"]
    abeta_plaque_df = abeta_plaque_df.dropna()

    data_frames = [individual_metadata_df, abeta_insoluble_cortex_df[{'individualID', 'cortex'}], abeta_insoluble_hip_df[{'individualID', 'hippocampus'}], abeta_plaque_df[{'individualID', 'plaque'}]]
    abeta_df = reduce(lambda left, right: pd.merge(left, right, on=['individualID'], how='outer'), data_frames)
    abeta_df = abeta_df.drop(abeta_df.loc[abeta_df['individualID']==572].index)  # exclude #572 from supervised samples because it should be an outlier.

    use_region = 1  # 0:cortex, 1:hippocampus, #2:plaque (ThioS+)
    if use_region == 0:
        use_region_name = 'cortex'
        abeta_df = abeta_df.drop(['hippocampus','plaque'], axis=1)
    elif use_region == 1:
        use_region_name = 'hippocampus'
        abeta_df = abeta_df.drop(['cortex','plaque'], axis=1)
    else:
        use_region_name = 'plaque'
        abeta_df = abeta_df.drop(['cortex','hippocampus'], axis=1)

    abeta_df = abeta_df.dropna()
    abeta_max = np.max(abeta_df[use_region_name][abeta_df['age'] == 12])
    abeta_mean_max = np.mean(abeta_df[use_region_name][abeta_df['age'] == 12])
    abeta_df[use_region_name] = abeta_df[use_region_name] / abeta_max

    abeta_df = abeta_df.reindex(columns=['individualID','age','type',use_region_name])

    abeta_scale_min = np.min(abeta_df.iloc[:, 3])
    for ind in range(10):
        abeta_df = abeta_df.append({'age': 8, 'type': 1, use_region_name: abeta_scale_min}, ignore_index=True)
        abeta_df = abeta_df.append({'age': 12, 'type': 1, use_region_name: abeta_scale_min}, ignore_index=True)
        abeta_df = abeta_df.append({'age': 18, 'type': 1, use_region_name: abeta_scale_min}, ignore_index=True)

    print("-------------------------------------")
    print("Loaded mouse abeta data:")
    print(abeta_df.columns)
    print(abeta_df)
    print("-------------------------------------")

    ##############################################################################
    # ****************************************************************************
    ##############################################################################
    # loading behavioral data
    open_field_df = pd.read_csv("./ScientificData_5xFAD/5xFAD_Open_Field.csv")
    elevated_plus_maze_df = pd.read_csv("./ScientificData_5xFAD/5xFAD_Elevated_Plus_Maze.csv")
    CFC_df = pd.read_csv("./ScientificData_5xFAD/5xFAD_CFC.csv")
    behavior_data_frames = [individual_metadata_df[{'individualID', 'age', 'type'}], open_field_df, elevated_plus_maze_df, CFC_df]
    behavior_merged_df = reduce(lambda left, right: pd.merge(left, right, on=['individualID'], how='outer'), behavior_data_frames)
    behavior_merged_df = behavior_merged_df.drop('Distance_cm', axis=1)
    behavior_merged_df['ratio in the center_s'] = behavior_merged_df['time in the center_s'] / behavior_merged_df['time in the arena_s']
    behavior_merged_df = behavior_merged_df.drop('time in the arena_s', axis=1)
    behavior_merged_df = behavior_merged_df.drop('time in the center_s', axis=1)
    behavior_merged_df = behavior_merged_df.reindex(columns=['individualID','age','type','Velocity_cm_s','ratio in the center_s','Cumulative open arms','Cumulative closed arms_seconds','Cumulative cente_seconds','baseline_train  activity within arena_mean','baseline_train  inactive freezing_frequency','baseline_train  inactive freezing_cumulative duration','test  activity within arena_mean','test inactive freezing_frequency','test  inactive freezing_cumulative duration'])

    original_data_df = behavior_merged_df.dropna()
    original_data_df = original_data_df.rename(columns={'ageDeath': 'age', 'individualCommonGenotype': 'type'})
    print("-------------------------------------")
    print("Loaded 5xFAD behavior data:")
    print(original_data_df.columns)
    print(original_data_df)
    print("-------------------------------------")

    #### scaling ######
    scaler = preprocessing.StandardScaler()
    original_data_df.iloc[:, 3:] = scaler.fit_transform(original_data_df.iloc[:, 3:])
    ######

    feature_num_array = np.array([1,4,7,10,11])
    replicates_per_featurenum = 5
    mse_Z_array = np.zeros([feature_num_array.shape[0], replicates_per_featurenum])

    for feature_num_ind, feature_num in enumerate(feature_num_array):
        for rep_ind in range(replicates_per_featurenum):
            data_df = original_data_df
            random_feature_index = np.random.permutation(np.arange(0,11,1))
            data_df = data_df.drop(data_df.columns[3+random_feature_index[feature_num:]], axis=1)

            data_frames = [data_df, abeta_df[{'individualID', use_region_name}]]
            both_data_df = reduce(lambda left, right: pd.merge(left, right, on=['individualID'], how='outer'), data_frames)
            both_data_df = both_data_df.dropna()
            print("-------------------------------------")
            print("Both abeta and behavior data:")
            print(both_data_df)
            print("-------------------------------------")

            sample_num = data_df.shape[0]
            data_df = data_df.fillna(data_df.median())

            t = np.int8(data_df['age'].values)
            s = np.int8(data_df['type'].values)
            X = data_df.iloc[:, 3:].values

            dim_X = X.shape[1]
            dim_s = np.unique(s).shape[0]

            # Pre Visualization ####################
            if feature_num != 1:
                pca = decomposition.PCA()
                pc_X = pca.fit_transform(X)
                figure_X_PC = plt.figure(figsize=(6, 6))
                plt.rcParams['font.size'] = '16'
                time_points = np.unique(t)
                for s_ind in range(dim_s):
                    for timing in time_points:
                        if s_ind == 0:
                            pp_color = 'r'
                        elif s_ind == 1:
                            pp_color = 'k'
                        if timing == 4:
                            pp_mark = 'o'
                        elif timing == 8:
                            pp_mark = '^'
                        elif timing == 12:
                            pp_mark = 'v'
                        elif timing == 18:
                            pp_mark = 's'
                        pp_format = pp_color + pp_mark
                        plt.plot(pc_X[np.logical_and(s == s_ind, t == timing), 0], pc_X[np.logical_and(s == s_ind, t == timing), 1],
                                 pp_format)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.savefig("PCscatter_featurenum"+str(feature_num)+"_rep"+str(rep_ind)+".eps")
                figure_X_PC.show()

            #####################################
            homedir_path = os.environ['HOME']

            Z_logprob_results = []
            unsupervisedZ_logprob_results = []
            testID_list = []

            testset_num = 4
            kf = KFold(n_splits=testset_num, shuffle=True, random_state=0)
            testset_ind = 0
            for train_index, test_index in kf.split(both_data_df):
                test_ID = both_data_df['individualID'].iloc[test_index]
                testID_list.append(test_ID)
                test_data_df = data_df.query('individualID in @test_ID')
                test_abeta_df = abeta_df.query('individualID in @test_ID')
                train_data_df = data_df.query('individualID not in @test_ID')
                train_abeta_df = abeta_df.query('individualID not in @test_ID')

                train_data_Zexist_samp_ind = []
                train_abeta_Xexist_samp_ind = []
                for abeta_samp_ind in range(train_abeta_df['individualID'].shape[0]):
                    tmp = np.where(train_data_df['individualID'].values == train_abeta_df['individualID'].values[abeta_samp_ind])[0]
                    if tmp.shape[0] > 0:
                        train_data_Zexist_samp_ind.append(tmp[0])
                        train_abeta_Xexist_samp_ind.append(abeta_samp_ind)
                train_Zexist_samp_num = len(train_data_Zexist_samp_ind)

                # config ############################
                dim_Z = 1
                dim_s = 2
                #####################################

                t = train_abeta_df['age'].values.astype(np.int8)
                s = train_abeta_df['type'].values.astype(np.int8)
                Z = train_abeta_df.iloc[:, 3:].values
                sample_num = train_abeta_df.shape[0]

                alpha = np.zeros([dim_s])
                beta = np.zeros([dim_s])
                q = np.zeros([dim_s])
                for s_ind in range(dim_s):
                    opt, cov = curve_fit(logistic, t[s==s_ind], Z[s==s_ind, 0], bounds=([0,-np.inf, 0],np.inf))
                    alpha[s_ind] = opt[0]
                    beta[s_ind] = opt[1]
                    q[s_ind] = opt[2]
                grad_Z_correction_factor = 4 / (alpha[0] * q[0])
                print("-------------------------")
                print("## alpha ##")
                print(alpha)
                print("## beta ##")
                print(beta)
                print("## q ##")
                print(q)
                print("-------------------------")

                nuts_kernel = NUTS(latent_model, init_strategy=initialization.init_to_sample())
                mcmc = MCMC(nuts_kernel, num_samples=3000, num_warmup=3000, num_chains=3)
                rng_key = random.PRNGKey(123)
                mcmc.run(rng_key, Z_obs=Z, t_obs=t, s_obs=s, mu_alpha_prior=alpha, mu_beta_prior=beta, mu_q_prior=q)
                mcmc.print_summary()
                # az.plot_trace(mcmc, var_names=["alpha[1,0]", "beta[1,0]"])
                posterior_samples = mcmc.get_samples()

                # compute moments of the pdfs.
                mu_alpha_prior = posterior_samples["mu_alpha"].mean(axis=0)
                mu_alpha_sigma_prior = posterior_samples["mu_alpha"].std(axis=0)
                prec_alpha_mean = posterior_samples["prec_alpha"].mean(axis=0)
                prec_alpha_std = posterior_samples["prec_alpha"].std(axis=0)
                prec_alpha_scale_prior = prec_alpha_mean / prec_alpha_std**2
                prec_alpha_shape_prior = prec_alpha_mean**2 / prec_alpha_std**2
                mu_beta_prior = posterior_samples["mu_beta"].mean(axis=0)
                mu_beta_sigma_prior = posterior_samples["mu_beta"].std(axis=0)
                prec_beta_mean = posterior_samples["prec_beta"].mean(axis=0)
                prec_beta_std = posterior_samples["prec_beta"].std(axis=0)
                prec_beta_scale_prior = prec_beta_mean / prec_beta_std ** 2
                prec_beta_shape_prior = prec_beta_mean ** 2 / prec_beta_std ** 2
                mu_q_prior = posterior_samples["mu_q"].mean(axis=0)
                mu_q_sigma_prior = posterior_samples["mu_q"].std(axis=0)
                prec_q_mean = posterior_samples["prec_q"].mean(axis=0)
                prec_q_std = posterior_samples["prec_q"].std(axis=0)
                prec_q_scale_prior = prec_q_mean / prec_q_std ** 2
                prec_q_shape_prior = prec_q_mean ** 2 / prec_q_std ** 2

                mu_alpha_mean = np.mean(posterior_samples["mu_alpha"], axis=0)
                mu_alpha_std = np.std(posterior_samples["mu_alpha"], axis=0)
                print("-------------------------")
                print("## alpha ##")
                print(mu_alpha_mean)
                mu_beta_mean = np.mean(posterior_samples["mu_beta"], axis=0)
                mu_beta_std = np.std(posterior_samples["mu_beta"], axis=0)
                print("## beta ##")
                print(mu_beta_mean)
                print("-------------------------")
                mu_q_mean = np.mean(posterior_samples["mu_q"], axis=0)
                mu_q_std = np.std(posterior_samples["mu_q"], axis=0)
                print("## q ##")
                print(mu_q_mean)
                print("-------------------------")

                fig_abeta = plt.figure(figsize=(5, 5))
                ax = []
                ax.append(fig_abeta.add_subplot(1, 1, 1))
                ax[0].plot(t[s == 0], Z[s == 0], 'ro')
                ax[0].plot(t[s == 1], Z[s == 1], 'ko')
                ax[0].set_xlim(0, 20)
                ax[0].set_xlabel("Age [m.o.]")
                ax[0].set_ylabel("Abeta")
                if use_region_name == 0:
                    ax[0].set_title("cortex")
                elif use_region_name == 1:
                    ax[0].set_title("hippocampus")
                fig_abeta.tight_layout()
                plt.savefig("original_abeta_timecourse_featurenum"+str(feature_num)+"_rep"+str(rep_ind)+ "_" + str(testset_ind) + ".eps")
                fig_abeta.show()

                fig_abeta_logistic = plt.figure(figsize=(5, 5))
                for s_ind in range(2):
                    mu_alpha_samp = norm.rvs(loc=mu_alpha_prior[s_ind], scale=mu_alpha_sigma_prior[s_ind], size=100)
                    prec_alpha_samp = gamma.rvs(prec_alpha_shape_prior[s_ind], loc=0, scale=prec_alpha_scale_prior[s_ind], size=100)
                    sigma_alpha_samp = 1 / np.sqrt(prec_alpha_samp)
                    mu_beta_samp = norm.rvs(loc=mu_beta_prior[s_ind], scale=mu_beta_sigma_prior[s_ind], size=100)
                    prec_beta_samp = gamma.rvs(prec_beta_shape_prior[s_ind], loc=0, scale=prec_beta_scale_prior[s_ind], size=100)
                    sigma_beta_samp = 1 / np.sqrt(prec_beta_samp)
                    mu_q_samp = norm.rvs(loc=mu_q_prior[s_ind], scale=mu_q_sigma_prior[s_ind], size=100)
                    prec_q_samp = gamma.rvs(prec_q_shape_prior[s_ind], loc=0, scale=prec_q_scale_prior[s_ind], size=100)
                    sigma_q_samp = 1 / np.sqrt(prec_q_samp)
                    for samp_ind in range(100):
                        alpha_samp = truncnorm.rvs((0.0 - mu_alpha_samp[samp_ind]) / sigma_alpha_samp[samp_ind], (np.inf - mu_alpha_samp[samp_ind]) / sigma_alpha_samp[samp_ind], loc=mu_alpha_samp[samp_ind], scale=sigma_alpha_samp[samp_ind], size=1)
                        beta_samp = norm.rvs(loc=mu_beta_samp[samp_ind], scale=sigma_beta_samp[samp_ind])
                        q_samp = truncnorm.rvs((0.0 - mu_q_samp[samp_ind]) / sigma_q_samp[samp_ind], (np.inf - mu_q_samp[samp_ind]) / sigma_q_samp[samp_ind], loc=mu_q_samp[samp_ind], scale=sigma_q_samp[samp_ind], size=1)
                        if s_ind == 0:
                            plt_color = 'r'
                        else:
                            plt_color = 'k'
                        plt.plot(np.arange(0, 25, 0.1), logistic(np.arange(0, 25, 0.1), alpha_samp, beta_samp, q_samp), plt_color, linewidth=1, alpha=0.3)
                plt.plot(t[s == 0], Z[s == 0], 'ro')
                plt.plot(t[s == 1], Z[s == 1], 'ko')
                plt.xlim(0, 20)
                plt.xlabel("Age [m.o.]")
                plt.ylabel("Abeta")
                if use_region == 0:
                    plt.title("cortex")
                elif use_region == 1:
                    plt.title("hippocampus")
                fig_abeta_logistic.tight_layout()
                plt.savefig("abeta_estimate_timecourse_featurenum"+str(feature_num)+"_rep"+str(rep_ind)+"_" + str(testset_ind) + ".eps")
                fig_abeta_logistic.show()

            ##########################

                train_t = np.int8(train_data_df['age'].values)
                train_s = np.int8(train_data_df['type'].values)
                train_X = train_data_df.iloc[:, 3:].values

                test_t = np.int8(test_data_df['age'].values)
                test_s = np.int8(test_data_df['type'].values)
                test_X = test_data_df.iloc[:, 3:].values
                test_Z = test_abeta_df[{use_region_name}].values

                train_data_noZexist_samp_ind = [ind for ind in range(train_t.shape[0]) if ind not in train_data_Zexist_samp_ind]
                unsupervised_t = train_t[train_data_noZexist_samp_ind]
                unsupervised_s = train_s[train_data_noZexist_samp_ind]
                unsupervised_X = train_X[train_data_noZexist_samp_ind]
                supervised_t = train_t[train_data_Zexist_samp_ind]
                supervised_s = train_s[train_data_Zexist_samp_ind]
                supervised_Z = train_abeta_df.iloc[train_abeta_Xexist_samp_ind, 3].values
                supervised_X = train_X[train_data_Zexist_samp_ind]
                X = np.concatenate([unsupervised_X, supervised_X])

                # config ############################
                nuts_samples = 5000
                nuts_warmup = 5000
                nuts_chains = 3
                #################################

                sample_num = X.shape[0]

                nuts_kernel = NUTS(full_model, init_strategy=initialization.init_to_sample())
                gibbs_fn = partial(_gibbs_fn_full, X)
                kernel = HMCGibbs(nuts_kernel, gibbs_fn=gibbs_fn, gibbs_sites=["W", "sigma_W", "sigma_X"])
                mcmc = MCMC(kernel, num_samples=nuts_samples, num_warmup=nuts_warmup, num_chains=nuts_chains)
                rng_key = random.PRNGKey(345)
                mcmc.run(rng_key, X_obs=unsupervised_X, t_obs=unsupervised_t, s_obs=unsupervised_s, supervised_X_obs=supervised_X, supervised_Z_obs=supervised_Z, supervised_t_obs=supervised_t, supervised_s_obs=supervised_s, mu_alpha_prior=mu_alpha_prior, mu_beta_prior=mu_beta_prior, mu_q_prior=mu_q_prior, mu_alpha_sigma=mu_alpha_sigma_prior, mu_beta_sigma=mu_beta_sigma_prior, mu_q_sigma=mu_q_sigma_prior, prec_alpha_shape=prec_alpha_shape_prior, prec_alpha_scale=prec_alpha_scale_prior, prec_beta_shape=prec_beta_shape_prior, prec_beta_scale=prec_beta_scale_prior, prec_q_shape=prec_q_shape_prior, prec_q_scale=prec_q_scale_prior, grad_Z_correction_factor=grad_Z_correction_factor)
                mcmc.print_summary()
                #az.plot_trace(mcmc, var_names=["alpha[1,0]", "beta[1,0]"])
                posterior_samples = mcmc.get_samples()

                alpha = np.mean(posterior_samples["alpha"], axis=0)
                beta = np.mean(posterior_samples["beta"], axis=0)
                q = np.mean(posterior_samples["q"], axis=0)
                mu_alpha = np.mean(posterior_samples["mu_alpha"], axis=0)
                mu_beta = np.mean(posterior_samples["mu_beta"], axis=0)
                mu_q = np.mean(posterior_samples["mu_q"], axis=0)
                prec_alpha = np.mean(posterior_samples["prec_alpha"], axis=0)
                prec_beta = np.mean(posterior_samples["prec_beta"], axis=0)
                prec_q = np.mean(posterior_samples["prec_q"], axis=0)
                W = np.mean(posterior_samples["W"], axis=0)
                sigma_W = np.mean(posterior_samples["sigma_W"], axis=0)
                sigma_X = np.mean(posterior_samples["sigma_X"], axis=0)
                Z_samp = np.mean(posterior_samples["Z"], axis=0)
                supervised_mu_Z_samp = np.mean(posterior_samples["supervised_mu_Z"], axis=0)
                grad_Z = np.mean(posterior_samples["Zo"], axis=0)[:, 1]

                #dill.dump_session("session_after_mcmc4real_testset"+str(testset_ind)+".pkl")

                #####################################################
                # prediction with the semi-supervised model.
                print('-------------------------')
                print('test Z')
                print(test_Z)
                print('-------------------------')

                log_prob = 0.0
                Z_points = np.arange(0,1.21,0.01)
                s_points = np.array([0, 1])
                t_points = np.arange(2, 19, 1)
                X_testsamp_num = test_X.shape[0]
                Z_logprob = np.zeros([X_testsamp_num, Z_points.shape[0]])
                for testsamp_ind in range(0, X_testsamp_num):
                    for Z_ind, new_Z in enumerate(Z_points):
                        print("------------------------------")
                        print(testsamp_ind)
                        print(new_Z)
                        print("------------------------------")
                        log_prob = predictive_Z(new_X_obs=test_X[testsamp_ind], new_Z_obs=new_Z, grad_Z_correction_factor=grad_Z_correction_factor, posterior_samples=posterior_samples)
                        Z_logprob[testsamp_ind, Z_ind] = log_prob
                Z_logprob_results.append(Z_logprob)

                testset_ind += 1

            # plot results of supervised prediction.
            # MAP
            total_test_samp_num = both_data_df.shape[0]
            predicted_Z = np.zeros([total_test_samp_num,])
            true_test_Z = np.zeros([total_test_samp_num, ])
            true_test_t = np.zeros([total_test_samp_num, ])
            Z_points = np.arange(0, 1.21, 0.01)
            cum_cnt = 0
            for testset_ind in range(testset_num):
                testsamp_num = len(Z_logprob_results[testset_ind])
                argmax_ind = np.argmax(Z_logprob_results[testset_ind],axis=1)
                predicted_Z[cum_cnt:cum_cnt + testsamp_num] = Z_points[argmax_ind]
                testset_str = 'testset' + str(testset_ind)
                test_ID = testID_list[testset_ind]
                test_abeta_df = abeta_df.query('individualID in @test_ID')
                true_test_Z[cum_cnt:cum_cnt + testsamp_num] = test_abeta_df.iloc[:, 3].values
                true_test_t[cum_cnt:cum_cnt + testsamp_num] = test_abeta_df['age'].values
                cum_cnt += testsamp_num
            mse_z_pred = mean_squared_error(true_test_Z, predicted_Z)
            mse_Z_array[feature_num_ind,rep_ind] = mse_z_pred

            fig_truetestZ_vs_predictedZ = plt.figure(figsize=(5,5))
            plt.rcParams['font.size'] = '16'
            plt.scatter(true_test_Z, predicted_Z)
            plt.plot(np.array([-0.05,1.05]),np.array([-0.05,1.05]),'k--', linewidth=1)
            mse_text = 'Mean squared error: ' + '{:.3f}'.format(mse_z_pred)
            plt.text(0.0, 1.15, mse_text, fontsize=16)
            #plt.text(0.0, 0.92, corr_text, fontsize=16)
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.xlabel('True abeta')
            plt.ylabel('Predicted abeta')
            plt.savefig("truetestZ_vs_predictedZ_MAP_featurenum"+str(feature_num)+"_rep"+str(rep_ind)+".eps")
            fig_truetestZ_vs_predictedZ.show()

            dill.dump_session("session_after_mcmc4real_featurenum" + str(feature_num) + "_rep" + str(rep_ind) + ".pkl")

    plt.figure(figsize=(5,4))
    sns.boxplot(np.repeat(feature_num_array, 5), np.reshape(mse_Z_array, mse_Z_array.shape[0] * mse_Z_array.shape[1]),color='C0', fliersize=0)
    sns.stripplot(np.repeat(feature_num_array, 5), np.reshape(mse_Z_array, mse_Z_array.shape[0] * mse_Z_array.shape[1]), color='k', size=4)
    plt.ylim([0, 0.5])
    plt.xlabel("The numbers of features")
    plt.ylabel("MSE")

    text = 'NumFeature 1 vs 11: ' + str(mannwhitneyu(mse_Z_array[0], mse_Z_array[4]))
    plt.text(0.0, 1.1, text, fontsize=3)
    text = 'NumFeature 4 vs 11: ' + str(mannwhitneyu(mse_Z_array[1], mse_Z_array[4]))
    plt.text(0.0, 1.07, text, fontsize=3)
    text = 'NumFeature 7 vs 11: ' + str(mannwhitneyu(mse_Z_array[2], mse_Z_array[4]))
    plt.text(0.0, 1.04, text, fontsize=3)
    text = 'NumFeature 1 vs 11: ' + str(mannwhitneyu(mse_Z_array[3], mse_Z_array[4]))
    plt.text(0.0, 1.01, text, fontsize=3)

    control = mse_Z_array[4]
    groups = [mse_Z_array[0], mse_Z_array[1], mse_Z_array[2], mse_Z_array[3]]
    p_values = []
    # Perform Mann-Whitney U-test
    for group in groups:
        _, p = mannwhitneyu(control, group, alternative='two-sided')  # Change the alternative if needed
        p_values.append(p)
    # Holm's correction
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.sort(p_values)
    holm_adjusted = sorted_p_values * (len(groups) - np.arange(len(groups)))
    holm_p = np.empty_like(holm_adjusted)
    holm_p[sorted_indices] = holm_adjusted
    print("Original p-values:", p_values)
    print("Holm-adjusted p-values:", holm_p)

    plt.savefig("Feature_num_vs_mse.eps")
    plt.show()

    dill.dump_session("session_after_mcmc4real_final.pkl")