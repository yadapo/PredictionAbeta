#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Written by Yuichiro Yada
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

################################################
#　A code for Fig. 2b-f, Supplementary Figure 1, 2
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
#plt.rcParams['figure.subplot.bottom'] = 0.15
#plt.rcParams['figure.subplot.left'] = 0.2
import matplotlib.cm as cm
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from scipy.optimize import curve_fit


from numpyro.infer import MCMC, NUTS, HMCGibbs, initialization
from scipy.stats import norm, truncnorm, gamma
from jax import random
import jax.numpy as jnp

from functools import partial, reduce

import dill
import arviz as az

from models import logistic, latent_model, full_model, _gibbs_fn_full, predictive_Z, predictive_s

if __name__ == "__main__":
    # loading individual data
    individual_metadata_df = pd.read_csv("./ScientificData_5xFAD/5xFAD_individual_metadata.csv")
    individual_metadata_df = individual_metadata_df[{"individualID", "ageDeath", "individualCommonGenotype"}]
    individual_metadata_df = individual_metadata_df.rename(columns={'ageDeath': 'age', 'individualCommonGenotype': 'type'})
    individual_metadata_df["type"].loc[individual_metadata_df["type"] == "5XFAD"] = 0
    individual_metadata_df["type"].loc[individual_metadata_df["type"] == "C57BL6J"] = 1

    abeta_type = "Abeta42"
    # loading mouse abeta data
    abeta_insoluble_df = pd.read_csv("./ScientificData_5xFAD/5xFAD_ab_levels insoluble_fraction.csv")
    abeta_insoluble_cortex_df = abeta_insoluble_df[abeta_insoluble_df["specimenID"].str.contains('cif')]
    abeta_insoluble_cortex_df = abeta_insoluble_cortex_df[{"individualID", "Abeta40", "Abeta42"}]
    abeta_insoluble_cortex_df["cortex"] = abeta_insoluble_cortex_df[abeta_type]
    abeta_insoluble_cortex_df = abeta_insoluble_cortex_df.dropna()

    abeta_insoluble_hip_df = abeta_insoluble_df[abeta_insoluble_df["specimenID"].str.contains('hif')]
    abeta_insoluble_hip_df = abeta_insoluble_hip_df[{"individualID", "Abeta40", "Abeta42"}]
    abeta_insoluble_hip_df["hippocampus"] = abeta_insoluble_hip_df[abeta_type]
    abeta_insoluble_hip_df = abeta_insoluble_hip_df.dropna()

    abeta_plaque_df = pd.read_csv("./ScientificData_5xFAD/5xFAD 4, 8 12 and 18 months_ ThioS-NeuN.csv")
    abeta_plaque_df = abeta_plaque_df[{"individualID","#objects/sqmm"}]
    abeta_plaque_df["plaque"] = abeta_plaque_df["#objects/sqmm"]
    abeta_plaque_df = abeta_plaque_df.dropna()

    data_frames = [individual_metadata_df, abeta_insoluble_cortex_df[{'individualID', 'cortex'}], abeta_insoluble_hip_df[{'individualID', 'hippocampus'}], abeta_plaque_df[{'individualID', 'plaque'}]]
    abeta_df = reduce(lambda left, right: pd.merge(left, right, on=['individualID'], how='outer'), data_frames)
    abeta_df = abeta_df.drop(abeta_df.loc[abeta_df['individualID']==572].index)  # exclude #572 from supervised samples because it should be an outlier.

    use_region = 0  # 0:cortex, 1:hippocampus, #2:plaque (ThioS+)
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

    data_df = behavior_merged_df.dropna()
    data_df = data_df.rename(columns={'ageDeath': 'age', 'individualCommonGenotype': 'type'})
    print("-------------------------------------")
    print("Loaded 5xFAD behavior data:")
    print(data_df.columns)
    print(data_df)
    print("-------------------------------------")

    #### scaling ######
    scaler = preprocessing.StandardScaler()
    data_df.iloc[:, 3:] = scaler.fit_transform(data_df.iloc[:, 3:])
    ######

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
    fig_xdata = plt.figure(figsize=(16, 14))
    plt.rcParams['font.size'] = '20'
    ax = []
    #feature_names = ['OF velocity','OF center ratio','EPM open arms','EPM closed arms','EPM center','CFC activity within arena pre','CFC freezing frequency pre','CFC freezing duration pre','CFC activity within arena post','CFC freezing frequency post','CFC freezing duration post']
    feature_names = ['OF-A', 'OF-B', 'EPM-A', 'EPM-B', 'EPM-C',
                     'CFC-A', 'CFC-B', 'CFC-C',
                     'CFC-D', 'CFC-E', 'CFC-F']
    for X_ind in range(dim_X):
        ax.append(fig_xdata.add_subplot(4, 4, X_ind + 1))
        ax[X_ind].tick_params(labelsize=24)
        ax[X_ind].plot(t[s == 0], X[s == 0, X_ind], 'ro')
        ax[X_ind].plot(t[s == 1], X[s == 1, X_ind], 'ko')
        ax[X_ind].set_xlim(0, 15)
        ax[X_ind].set_xticks([0, 4, 8, 12])
        ax[X_ind].set_xlabel("Age [m.o.]", fontsize=24)
        ax[X_ind].set_ylabel(feature_names[X_ind], fontsize=24)
    fig_xdata.tight_layout()
    plt.savefig("RawXtimecourse.eps")
    fig_xdata.show()

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
    plt.savefig("PCscatter.eps")
    figure_X_PC.show()

    figure_X_PC_timecourse = plt.figure(figsize=(4, 6))
    ax = []
    for pc_ind in range(3):
        ax.append(figure_X_PC_timecourse.add_subplot(3, 1, pc_ind + 1))
        ax[pc_ind].plot(t[s == 0], pc_X[s == 0, pc_ind], 'ro')
        ax[pc_ind].plot(t[s == 1], pc_X[s == 1, pc_ind], 'ko')
        ax[pc_ind].set_xlim(2, 14)
        ax[pc_ind].set_xticks([4, 8, 12])
        ax[pc_ind].set_xlabel("Age [m.o.]")
        ax[pc_ind].set_ylabel("PC" + str(pc_ind + 1))
    figure_X_PC_timecourse.tight_layout()
    plt.savefig("PCtimecourse.eps")
    figure_X_PC_timecourse.show()

    unsupervised_X = data_df[~data_df['individualID'].isin(both_data_df['individualID'])].iloc[:, 3:].values
    unsupervised_t = data_df['age'][~data_df['individualID'].isin(both_data_df['individualID'])].values
    unsupervised_s = data_df['type'][~data_df['individualID'].isin(both_data_df['individualID'])].values
    supervised_X = data_df[data_df['individualID'].isin(both_data_df['individualID'])].iloc[:, 3:].values
    supervised_t = data_df['age'][data_df['individualID'].isin(both_data_df['individualID'])].values
    supervised_s = data_df['type'][data_df['individualID'].isin(both_data_df['individualID'])].values
    pca = decomposition.PCA()
    pca.fit_transform(X)
    pc_X_unsupervised = pca.transform(unsupervised_X)
    pc_X_supervised = pca.transform(supervised_X)
    figure_X_PC_su = plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = '20'
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
            plt.plot(pc_X_unsupervised[np.logical_and(unsupervised_s == s_ind, unsupervised_t == timing), 0],
                     pc_X_unsupervised[np.logical_and(unsupervised_s == s_ind, unsupervised_t == timing), 1], pp_format)
    for s_ind in range(dim_s):
        for timing in time_points:
            if s_ind == 0:
                pp_color = 'm'
            elif s_ind == 1:
                pp_color = 'c'
            if timing == 4:
                pp_mark = 'o'
            elif timing == 8:
                pp_mark = '^'
            elif timing == 12:
                pp_mark = 'v'
            elif timing == 18:
                pp_mark = 's'
            pp_format = pp_color + pp_mark
            plt.plot(pc_X_supervised[np.logical_and(supervised_s == s_ind, supervised_t == timing), 0],
                     pc_X_supervised[np.logical_and(supervised_s == s_ind, supervised_t == timing), 1], pp_format)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig("PCscatter_supervised_unsupervised.eps")
    figure_X_PC_su.show()

    both_X = both_data_df.iloc[:, 3:3+dim_X].values
    both_Z = both_data_df.iloc[:, 3+dim_X:].values
    both_s = both_data_df["type"].values
    both_t = both_data_df["age"].values
    pc_X = pca.transform(both_X)
    figure_X_PC_vsZ = plt.figure(figsize=(5, 5))
    ax = []
    for pc_ind in range(6):
        ax.append(figure_X_PC_vsZ.add_subplot(3, 2, pc_ind + 1))
        ax[pc_ind].plot(both_Z[both_s ==0], pc_X[both_s == 0, pc_ind], 'ro')
        ax[pc_ind].set_xlabel("Abeta")
        ax[pc_ind].set_ylabel("PC" + str(pc_ind + 1))
    figure_X_PC_vsZ.tight_layout()
    plt.savefig("PCvsZ.eps")
    figure_X_PC_vsZ.show()

    figure_X_PC_vsZ = plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = '16'
    for samp_ind in range(both_Z.shape[0]):
        plt.scatter(pc_X[samp_ind, 0], pc_X[samp_ind, 1], color=cm.hsv(both_Z[samp_ind]))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig("PCvsZ_scatter.eps")
    figure_X_PC_vsZ.show()


    #####################################
    homedir_path = os.environ['HOME']

    Z_logprob_results = []
    s_logprob_results = []
    testID_list = []

    loo = LeaveOneOut()
    testset_ind = 0
    for train_index, test_index in loo.split(both_data_df):
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
        plt.savefig("original_abeta_timecourse" + str(testset_ind) + ".eps")
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
        plt.savefig("abeta_estimate_timecourse" + str(testset_ind) + ".eps")
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

        dill.dump_session("session_after_mcmc4real_testset"+str(testset_ind)+".pkl")

        fig_trueZ_vs_latentZ = plt.figure(figsize=(5, 5))
        plt.rcParams['font.size'] = '16'
        plt.scatter(train_abeta_df.iloc[train_abeta_Xexist_samp_ind,3].values, supervised_mu_Z_samp[0])
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('True abeta')
        plt.ylabel('Latent abeta')
        plt.savefig("trueZ_vs_latentZ"+str(testset_ind)+".eps")
        fig_trueZ_vs_latentZ.show()

        df = pd.DataFrame([], columns=['type', 'Z'])
        df['type'] = unsupervised_s
        df['Z'] = Z_samp[0]
        fig_latentZ_wt_vs_tg = plt.figure(figsize=(5, 5))
        plt.rcParams['font.size'] = '16'
        sns.catplot(x='type', y='Z', data=df, kind='swarm')
        plt.ylim(-0.05,1.05)
        plt.savefig("latentZ_wt_vs_tg"+str(testset_ind)+".eps")
        fig_latentZ_wt_vs_tg.show()

        fig_latentZ_timecourse = plt.figure(figsize=(5, 5))
        plt.rcParams['font.size'] = '16'
        plt.plot(unsupervised_t[unsupervised_s == 0], Z_samp[0][unsupervised_s == 0], 'ro')
        plt.plot(unsupervised_t[unsupervised_s == 1], Z_samp[0][unsupervised_s == 1], 'ko')
        plt.ylim(-0.05,1.05)
        plt.xlabel('Age [m.o.]')
        plt.ylabel('Latent abeta')
        plt.savefig("latentZ_timecourse"+str(testset_ind)+".eps")
        fig_latentZ_timecourse.show()

        fig_gradZ_timecourse = plt.figure(figsize=(5, 5))
        plt.rcParams['font.size'] = '16'
        plt.plot(unsupervised_t[unsupervised_s == 0], grad_Z[unsupervised_s == 0], 'ro')
        plt.plot(unsupervised_t[unsupervised_s == 1], grad_Z[unsupervised_s == 1], 'ko')
        # plt.ylim(-0.05,-)
        plt.xlabel('Age [m.o.]')
        plt.ylabel('Grad abeta')
        plt.savefig("gradZ_timecourse" + str(testset_ind) + ".eps")
        fig_gradZ_timecourse.show()

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

        fig_test_Z_estimate = plt.figure(figsize=(6, 4))
        ax = []
        for testsamp_ind in range(X_testsamp_num):
            ax.append(fig_test_Z_estimate.add_subplot(np.int(np.ceil(X_testsamp_num / 4)), 4, testsamp_ind+ 1))
            ax[testsamp_ind].plot(Z_points, Z_logprob[testsamp_ind, :], 'ro', markersize=2)
            ax[testsamp_ind].set_xlabel("Z")
            ax[testsamp_ind].set_ylabel("log(Density)")
        fig_test_Z_estimate.tight_layout()
        plt.savefig("rawX_Z_estimate_testset"+str(testset_ind)+".eps")
        fig_test_Z_estimate.show()

        # predicting the type and age of mice ################################
        log_prob = 0.0
        s_points = np.array([0, 1])
        t_points = np.arange(2, 19, 1)
        X_testsamp_num = test_X.shape[0]
        s_logprob = np.zeros([X_testsamp_num, s_points.shape[0]])
        for testsamp_ind in range(0, X_testsamp_num):
            for s_ind, new_s in enumerate(s_points):
                print("------------------------------")
                print(testsamp_ind)
                print('type' + str(new_s))
                print("------------------------------")
                log_prob = predictive_s(new_X_obs=test_X[testsamp_ind], new_s_obs=new_s,
                                        grad_Z_correction_factor=grad_Z_correction_factor,
                                        posterior_samples=posterior_samples)
                s_logprob[testsamp_ind, s_ind] = log_prob
        s_logprob_results.append(s_logprob)

        dill.dump_session("session_after_mcmc4real_testset"+str(testset_ind)+".pkl")
        testset_ind += 1

    # plot results of supervised prediction.
    # MAP
    total_test_samp_num = len(testID_list)
    predicted_Z = np.zeros([total_test_samp_num,])
    true_test_Z = np.zeros([total_test_samp_num, ])
    true_test_t = np.zeros([total_test_samp_num, ])
    Z_points = np.arange(0, 1.21, 0.01)
    cum_cnt = 0
    for testset_ind in range(total_test_samp_num):
        argmax_ind = np.argmax(Z_logprob_results[testset_ind],axis=1)
        predicted_Z[testset_ind] = Z_points[argmax_ind]
        testset_str = 'testset' + str(testset_ind)
        test_ID = testID_list[testset_ind]
        test_abeta_df = abeta_df.query('individualID in @test_ID')
        true_test_Z[testset_ind] = test_abeta_df.iloc[:, 3].values
        true_test_t[testset_ind] = test_abeta_df['age'].values
    mse_z_pred = mean_squared_error(true_test_Z, predicted_Z)
    z_corrcoef = np.corrcoef(true_test_Z, predicted_Z)

    fig_truetestZ_vs_predictedZ = plt.figure(figsize=(5,5))
    plt.rcParams['font.size'] = '16'
    plt.scatter(true_test_Z, predicted_Z)
    plt.plot(np.array([-0.05,1.05]),np.array([-0.05,1.05]),'k--', linewidth=1)
    mse_text = 'Mean squared error: ' + '{:.3f}'.format(mse_z_pred)
    corr_text = 'Correlation coef.: ' + '{:.3f}'.format(z_corrcoef[1,0])
    plt.text(0.0, 1.15, mse_text, fontsize=16)
    #plt.text(0.0, 0.92, corr_text, fontsize=16)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('True abeta')
    plt.ylabel('Predicted abeta')
    plt.savefig("truetestZ_vs_predictedZ_MAP.eps")
    fig_truetestZ_vs_predictedZ.show()

    # plot predicted z only on particular day samples
    fig_truetestZ_vs_predictedZ = plt.figure(figsize=(5, 5))
    plt.plot(np.array([-0.05, 1.05]), np.array([-0.05, 1.05]), 'k--', linewidth=1)
    mse_z_pred = mean_squared_error(true_test_Z, predicted_Z)
    mse_text = 'Mean squared error: ' + '{:.3f}'.format(mse_z_pred)
    for day_ind, selected_day in enumerate(np.array([8,12])):
        mse_z_pred = mean_squared_error(true_test_Z[true_test_t==selected_day], predicted_Z[true_test_t==selected_day])
        mse_text = 'Mean squared error: ' + '{:.3f}'.format(mse_z_pred)
        if selected_day == 8:
            color = 'c'
            plt.text(0.0, 1.10, mse_text, fontsize=16)
        elif selected_day == 12:
            color = 'b'
            plt.text(0.0, 1.15, mse_text, fontsize=16)
        plt.scatter(true_test_Z[true_test_t==selected_day], predicted_Z[true_test_t==selected_day], c=color)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('True abeta')
    plt.ylabel('Predicted abeta')
    plt.savefig("truetestZ_vs_predictedZ_day_MAP_given_ts.eps")
    fig_truetestZ_vs_predictedZ.show()

    # plot results of supervised prediction with type prediction results. ##########################
    predicted_s = np.zeros([total_test_samp_num, ])
    true_test_s = np.zeros([total_test_samp_num, ])
    s_points = np.array([0, 1])
    cum_cnt = 0
    for testset_ind in range(total_test_samp_num):
        s_tmax_ll = np.zeros(s_points.shape[0])
        set_ll_results = s_logprob_results[testset_ind]
        samp_ll_results = set_ll_results[0]
        s_argmax_ind = np.argmax(samp_ll_results)
        predicted_s[testset_ind] = s_points[s_argmax_ind]
        argmax_ind = np.argmax(Z_logprob_results[testset_ind], axis=1)
        predicted_Z[testset_ind] = Z_points[argmax_ind]
        testset_str = 'testset' + str(testset_ind)
        test_ID = testID_list[testset_ind]
        test_abeta_df = abeta_df.query('individualID in @test_ID')
        true_test_Z[testset_ind] = test_abeta_df.iloc[:, 3].values
        true_test_t[testset_ind] = test_abeta_df['age'].values

    fig_truetestZ_vs_predictedZ_wtype = plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = '16'
    for samp_ind in range(total_test_samp_num):
        if predicted_s[samp_ind]:
            color = 'b'
        else:
            color = 'r'
        plt.scatter(true_test_Z[samp_ind], predicted_Z[samp_ind], color=color)
    plt.plot(np.array([-0.05, 1.05]), np.array([-0.05, 1.05]), 'k--', linewidth=1)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('True abeta')
    plt.ylabel('Predicted abeta')
    plt.savefig("truetestZ_vs_predictedZ_withtype_MAP.eps")
    fig_truetestZ_vs_predictedZ_wtype.show()

    dill.dump_session("session_after_mcmc4real_final.pkl")