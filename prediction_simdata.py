#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Written by Yuichiro Yada
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

################################################
#　A code for Fig. 5b-d
################################################


import numpy as np
import pandas as pd
import seaborn as sns
import os

import matplotlib.pyplot as plt
import sklearn.model_selection

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
from sklearn import decomposition
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

from numpyro.infer import MCMC, NUTS, HMCGibbs, initialization
from scipy.stats import norm, truncnorm, gamma
from jax import random

from functools import partial

import dill
import arviz as az

from models import logistic, latent_model, full_model, _gibbs_fn_full, predictive_Z, simulate_data

if __name__ == "__main__":
    # config ############################
    dim_X = 10
    dim_Z = 1
    dim_s = 2
    #####################################

    sample_num = 240

    t = np.random.randint(low=2, high=9,  size=sample_num) * 2
    s = np.int8(np.concatenate((np.zeros([int(sample_num/2),]), np.ones([int(sample_num/2),]))))

    alpha = np.array([0.8, 0.3])
    beta = np.array([-5.0, -15.0])
    q = np.array([0.6, 0.001])
    mu_alpha_sigma_prior = 0.1 * np.ones(2)
    mu_beta_sigma_prior = 1 * np.ones(2)
    mu_q_sigma_prior = 0.01 * np.ones(2)
    prec_alpha_shape_prior = np.array([100, 100])
    prec_alpha_scale_prior = np.array([0.5, 1])
    prec_beta_shape_prior = 100 * np.ones(2)
    prec_beta_scale_prior = 1 * np.ones(2)
    prec_q_shape_prior = np.array([25, 100])
    prec_q_scale_prior = np.array([1, 1])
    grad_Z_correction_factor = 4 / (alpha[0] * q[0])

    sigma_X = np.random.gamma(10, 0.1, dim_X)
    X, Z, t, s, alpha, beta, q, W, sigma_W, sigma_X, mu_alpha, sigma_alpha, mu_beta, sigma_beta, mu_q, sigma_q = simulate_data(
        random.PRNGKey(10), sample_num, dim_X, dim_Z, dim_s, t=t, s=s, mu_alpha_prior=alpha,
        mu_beta_prior=beta, mu_q_prior=q, mu_alpha_sigma=mu_alpha_sigma_prior,
        mu_beta_sigma=mu_beta_sigma_prior, mu_q_sigma=mu_q_sigma_prior, prec_alpha_shape=prec_alpha_shape_prior,
        prec_alpha_scale=prec_alpha_scale_prior, prec_beta_shape=prec_beta_shape_prior,
        prec_beta_scale=prec_beta_scale_prior, prec_q_shape=prec_q_shape_prior, prec_q_scale=prec_q_scale_prior,
        sigma_X=sigma_X,
        grad_Z_correction_factor=grad_Z_correction_factor, wt_z_zero=1)
    trueW = W

    fig_latentZ_timecourse = plt.figure(figsize=(4, 4))
    plt.rcParams['font.size'] = '16'
    plt.plot(t[s == 0], Z[s == 0], 'ro')
    plt.plot(t[s == 1], Z[s == 1], 'ko')
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Age [m.o.]')
    plt.ylabel('Latent abeta')
    plt.savefig("generated_simdata_latentZ_timecourse.eps")
    fig_latentZ_timecourse.show()

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
    plt.savefig("generated_simdata_X_PCscatter.eps")
    figure_X_PC.show()

    abeta_only_observed_num = 40
    unsupervised_num_array = np.array([176, 160, 120, 100, 80])

    for unsupervised_num in unsupervised_num_array:
        supervised_num = sample_num - (abeta_only_observed_num + unsupervised_num)
        [abeta_only_observed_ind, unsupervised_ind, supervised_ind] = np.split(np.random.permutation(np.arange(sample_num)), [abeta_only_observed_num, abeta_only_observed_num+unsupervised_num])

        #####################################
        testset_num = 4
        input = np.stack([t,s], axis=1)
        input_label = s * 100 + t
        kf = sklearn.model_selection.StratifiedKFold(n_splits=testset_num, shuffle=True)

        abeta_only_observed_t = t[abeta_only_observed_ind]
        abeta_only_observed_s = s[abeta_only_observed_ind]
        abeta_only_observed_Z = np.expand_dims(Z[abeta_only_observed_ind], axis=1)
        abeta_only_observed_X = X[abeta_only_observed_ind]
        unsupervised_t = t[unsupervised_ind]
        unsupervised_s = s[unsupervised_ind]
        unsupervised_Z = np.expand_dims(Z[unsupervised_ind], axis=1)
        unsupervised_X = X[unsupervised_ind]
        supervised_t = t[supervised_ind]
        supervised_s = s[supervised_ind]
        supervised_Z = np.expand_dims(Z[supervised_ind], axis=1)
        supervised_X = X[supervised_ind]

        Z_logprob_results = []
        testset_ind = 0
        for train_index, test_index in kf.split(X[supervised_ind], input_label[supervised_ind]):

            supervised_train_X, supervised_test_X = supervised_X[train_index], supervised_X[test_index]
            supervised_train_Z, supervised_test_Z = supervised_Z[train_index], supervised_Z[test_index]
            supervised_train_t, supervised_test_t = supervised_t[train_index], supervised_t[test_index]
            supervised_train_s, supervised_test_s = supervised_s[train_index], supervised_s[test_index]
            trainsamp_num = train_index.shape

            supervised_train_num = supervised_train_t.shape[0]

            train_t = np.concatenate([unsupervised_t, supervised_train_t])
            train_s = np.concatenate([unsupervised_s, supervised_train_s])
            train_X = np.concatenate([unsupervised_X, supervised_train_X])

            abeta_t = np.concatenate([abeta_only_observed_t, supervised_train_t])
            abeta_s = np.concatenate([abeta_only_observed_s, supervised_train_s])
            abeta_Z = np.concatenate([abeta_only_observed_Z, supervised_train_Z])

            #####################################
            alpha = np.zeros([dim_s])
            beta = np.zeros([dim_s])
            q = np.zeros([dim_s])
            for s_ind in range(dim_s):
                opt, cov = curve_fit(logistic, abeta_t[abeta_s == s_ind], abeta_Z[abeta_s == s_ind,0], bounds=([0, -np.inf, 0], np.inf))
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

            # config ############################
            nuts_samples = 3000
            nuts_warmup = 3000
            nuts_chains = 3
            #################################

            nuts_kernel = NUTS(latent_model, init_strategy=initialization.init_to_sample())
            mcmc = MCMC(nuts_kernel, num_samples=nuts_samples, num_warmup=nuts_warmup, num_chains=nuts_chains)
            rng_key = random.PRNGKey(123)
            mcmc.run(rng_key, Z_obs=abeta_Z, t_obs=abeta_t, s_obs=abeta_s, mu_alpha_prior=alpha, mu_beta_prior=beta, mu_q_prior=q)
            mcmc.print_summary()
            # az.plot_trace(mcmc, var_names=["alpha[1,0]", "beta[1,0]"])
            posterior_samples = mcmc.get_samples()

            # compute moments of the pdfs.
            mu_alpha_prior = posterior_samples["mu_alpha"].mean(axis=0)
            mu_alpha_sigma_prior = posterior_samples["mu_alpha"].std(axis=0)
            prec_alpha_mean = posterior_samples["prec_alpha"].mean(axis=0)
            prec_alpha_std = posterior_samples["prec_alpha"].std(axis=0)
            prec_alpha_scale_prior = prec_alpha_mean / prec_alpha_std ** 2
            prec_alpha_shape_prior = prec_alpha_mean ** 2 / prec_alpha_std ** 2
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

            #####################################
            # Plot sample logistic curves

            fig_abeta = plt.figure(figsize=(10, 6))
            ax = []
            for Z_ind in range(dim_Z):
                ax.append(fig_abeta.add_subplot(1, dim_Z, Z_ind + 1))
                ax[Z_ind].plot(abeta_t[abeta_s == 0], abeta_Z[abeta_s == 0, Z_ind], 'ro')
                ax[Z_ind].plot(abeta_t[abeta_s == 1], abeta_Z[abeta_s == 1, Z_ind], 'ko')
                ax[Z_ind].set_xlim(0, 20)
                # ax[Z_ind].set_ylim(-0.05, 1.05)
                ax[Z_ind].set_xlabel("Age [m.o.]")
                ax[Z_ind].set_ylabel("Abeta")
            fig_abeta.tight_layout()
            fig_abeta.show()

            fig_abeta_logistic = plt.figure(figsize=(5, 5))
            for s_ind in range(2):
                mu_alpha_samp = norm.rvs(loc=mu_alpha_prior[s_ind], scale=mu_alpha_sigma_prior[s_ind], size=100)
                prec_alpha_samp = gamma.rvs(prec_alpha_shape_prior[s_ind], loc=0, scale=prec_alpha_scale_prior[s_ind],
                                            size=100)
                sigma_alpha_samp = 1 / np.sqrt(prec_alpha_samp)
                mu_beta_samp = norm.rvs(loc=mu_beta_prior[s_ind], scale=mu_beta_sigma_prior[s_ind], size=100)
                prec_beta_samp = gamma.rvs(prec_beta_shape_prior[s_ind], loc=0, scale=prec_beta_scale_prior[s_ind],
                                           size=100)
                sigma_beta_samp = 1 / np.sqrt(prec_beta_samp)
                mu_q_samp = norm.rvs(loc=mu_q_prior[s_ind], scale=mu_q_sigma_prior[s_ind], size=100)
                prec_q_samp = gamma.rvs(prec_q_shape_prior[s_ind], loc=0, scale=prec_q_scale_prior[s_ind], size=100)
                sigma_q_samp = 1 / np.sqrt(prec_q_samp)
                for samp_ind in range(100):
                    alpha_samp = truncnorm.rvs((0.0 - mu_alpha_samp[samp_ind]) / sigma_alpha_samp[samp_ind],
                                               (np.inf - mu_alpha_samp[samp_ind]) / sigma_alpha_samp[samp_ind],
                                               loc=mu_alpha_samp[samp_ind], scale=sigma_alpha_samp[samp_ind], size=1)
                    beta_samp = norm.rvs(loc=mu_beta_samp[samp_ind], scale=sigma_beta_samp[samp_ind])
                    q_samp = truncnorm.rvs((0.0 - mu_q_samp[samp_ind]) / sigma_q_samp[samp_ind],
                                           (np.inf - mu_q_samp[samp_ind]) / sigma_q_samp[samp_ind], loc=mu_q_samp[samp_ind],
                                           scale=sigma_q_samp[samp_ind], size=1)
                    if s_ind == 0:
                        plt_color = 'r'
                    else:
                        plt_color = 'k'
                    plt.plot(np.arange(0, 25, 0.1), logistic(np.arange(0, 25, 0.1), alpha_samp, beta_samp, q_samp),
                             plt_color,
                             linewidth=1, alpha=0.3)
            plt.plot(abeta_t[abeta_s == 0], abeta_Z[abeta_s == 0], 'ro')
            plt.plot(abeta_t[abeta_s == 1], abeta_Z[abeta_s == 1], 'ko')
            plt.xlim(0, 20)
            plt.xlabel("Age [m.o.]")
            plt.ylabel("Abeta")
            fig_abeta_logistic.tight_layout()
            plt.savefig("abeta_estimate_timecourse_unsupervised_num_"+str(unsupervised_num)+"testset"+str(testset_ind)+".png")
            fig_abeta_logistic.show()

            # config ############################
            nuts_samples = 5000
            nuts_warmup = 5000
            nuts_chains = 3
            #################################

            supervised_train_Z = np.squeeze(supervised_train_Z)
            unsupervised_Z = np.squeeze(unsupervised_Z)

            nuts_kernel = NUTS(full_model, init_strategy=initialization.init_to_sample())
            gibbs_fn_full = partial(_gibbs_fn_full, train_X)
            kernel = HMCGibbs(nuts_kernel, gibbs_fn=gibbs_fn_full, gibbs_sites=["W", "sigma_W", "sigma_X"])
            mcmc = MCMC(kernel, num_samples=nuts_samples, num_warmup=nuts_warmup, num_chains=nuts_chains)
            rng_key = random.PRNGKey(122)
            mcmc.run(rng_key, X_obs=unsupervised_X, t_obs=unsupervised_t, s_obs=unsupervised_s, supervised_X_obs=supervised_train_X, supervised_Z_obs=supervised_train_Z, supervised_t_obs=supervised_train_t, supervised_s_obs=supervised_train_s, mu_alpha_prior=mu_alpha_prior, mu_beta_prior=mu_beta_prior, mu_q_prior=mu_q_prior, mu_alpha_sigma=mu_alpha_sigma_prior, mu_beta_sigma=mu_beta_sigma_prior, mu_q_sigma=mu_q_sigma_prior, prec_alpha_shape=prec_alpha_shape_prior, prec_alpha_scale=prec_alpha_scale_prior, prec_beta_shape=prec_beta_shape_prior, prec_beta_scale=prec_beta_scale_prior, prec_q_shape=prec_q_shape_prior, prec_q_scale=prec_q_scale_prior, grad_Z_correction_factor=grad_Z_correction_factor)
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
            Z_samp = np.mean(posterior_samples["Z"], axis=0)
            supervised_mu_Z_samp = np.mean(posterior_samples["supervised_mu_Z"], axis=0)
            grad_Z = np.mean(posterior_samples["Zo"], axis=0)[:, 1]

            dill.dump_session("session_after_mcmc4real_unsupervised_num_"+str(unsupervised_num)+"testset"+str(testset_ind)+".pkl")

            ## visualize true and posterior W
            figure_W_heatmap = plt.figure(figsize=(5,5))
            ax = []
            ax.append(figure_W_heatmap.add_subplot(1, 2, 1))
            ax[0].imshow(trueW, vmin=-5, vmax=5)
            ax[0].set_title("true W")
            ax.append(figure_W_heatmap.add_subplot(1, 2, 2))
            shw = ax[1].imshow(W, vmin=-5, vmax=5)
            ax[1].set_title("mean sampled W")
            bar = plt.colorbar(shw)
            plt.savefig("W_heatmap_unsupervised_num_" + str(unsupervised_num) + "testset" + str(testset_ind) + ".eps")
            figure_W_heatmap.show()

            fig_trueZ_vs_latentZ = plt.figure(figsize=(5,5))
            plt.rcParams['font.size'] = '16'
            plt.scatter(supervised_train_Z, supervised_mu_Z_samp[0])
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.xlabel('True abeta')
            plt.ylabel('Latent abeta')
            plt.savefig("trueZ_vs_latentZ_unsupervised_num_"+str(unsupervised_num)+"testset"+str(testset_ind)+".eps")
            fig_trueZ_vs_latentZ.show()

            df = pd.DataFrame([], columns=['type', 'Z'])
            df['type'] = unsupervised_s
            df['Z'] = Z_samp[0]
            fig_latentZ_wt_vs_tg = plt.figure(figsize=(5,5))
            plt.rcParams['font.size'] = '16'
            sns.catplot(x='type', y='Z', data=df, kind='swarm')
            plt.ylim(-0.05,1.05)
            plt.savefig("latentZ_wt_vs_tg_unsupervised_num_"+str(unsupervised_num)+"testset"+str(testset_ind)+".eps")
            fig_latentZ_wt_vs_tg.show()

            fig_latentZ_timecourse = plt.figure(figsize=(5,5))
            plt.rcParams['font.size'] = '16'
            plt.plot(unsupervised_t[unsupervised_s == 0], Z_samp[0][unsupervised_s == 0], 'ro')
            plt.plot(unsupervised_t[unsupervised_s == 1], Z_samp[0][unsupervised_s == 1], 'ko')
            plt.ylim(-0.05,1.05)
            plt.xlabel('Age [m.o.]')
            plt.ylabel('Latent abeta')
            plt.savefig("latentZ_timecourse_unsupervised_num_"+str(unsupervised_num)+"testset"+str(testset_ind)+".eps")
            fig_latentZ_timecourse.show()

            fig_gradZ_timecourse = plt.figure(figsize=(5,5))
            plt.rcParams['font.size'] = '16'
            plt.plot(unsupervised_t[unsupervised_s == 0], grad_Z[unsupervised_s == 0], 'ro')
            plt.plot(unsupervised_t[unsupervised_s == 1], grad_Z[unsupervised_s == 1], 'ko')
            #plt.ylim(-0.05,-)
            plt.xlabel('Age [m.o.]')
            plt.ylabel('Grad abeta')
            plt.savefig("gradZ_timecourse_unsupervised_num_"+str(unsupervised_num)+"testset"+str(testset_ind)+".eps")
            fig_gradZ_timecourse.show()

            print('-------------------------')
            print('test Z')
            print(supervised_test_Z)
            print('-------------------------')

            Z_points = np.arange(0,1.21,0.01)
            s_points = np.array([0, 1])
            t_points = np.arange(2, 19, 1)
            X_testsamp_num = supervised_test_X.shape[0]
            Z_logprob = np.zeros([X_testsamp_num, Z_points.shape[0]])
            for testsamp_ind in range(0, X_testsamp_num):
                for Z_ind, Z_pred in enumerate(Z_points):
                    print("------------------------------")
                    print(testsamp_ind)
                    print(Z_pred)
                    print("------------------------------")
                    log_prob = predictive_Z(new_X_obs=supervised_test_X[testsamp_ind], new_Z_obs=Z_pred, grad_Z_correction_factor=grad_Z_correction_factor, posterior_samples=posterior_samples)
                    Z_logprob[testsamp_ind, Z_ind] = log_prob
            Z_logprob_results.append(Z_logprob)

            dill.dump_session("session_after_mcmc4real_unsupervised_num_"+str(unsupervised_num)+"testset"+str(testset_ind)+".pkl")

            fig_test_Z_estimate = plt.figure(figsize=(6, 4))
            ax = []
            for testsamp_ind in range(X_testsamp_num):
                ax.append(fig_test_Z_estimate.add_subplot(np.int(np.ceil(X_testsamp_num / 4)), 4, testsamp_ind+ 1))
                ax[testsamp_ind].plot(Z_points, Z_logprob[testsamp_ind, :], 'ro', markersize=2)
                ax[testsamp_ind].set_xlabel("Z")
                ax[testsamp_ind].set_ylabel("log(Density)")
            fig_test_Z_estimate.tight_layout()
            plt.savefig("rawX_Z_estimate_unsupervised_num_"+str(unsupervised_num)+"testset"+str(testset_ind)+".png")
            fig_test_Z_estimate.show()

            dill.dump_session("session_after_mcmc4real_unsupervised_num_"+str(unsupervised_num)+"testset"+str(testset_ind)+".pkl")

            testset_ind += 1

        total_test_samp_num = sample_num - abeta_only_observed_num - unsupervised_num
        predicted_Z = np.zeros([total_test_samp_num,])
        true_test_Z = np.zeros([total_test_samp_num,])
        true_test_t = np.zeros([total_test_samp_num,])
        dill.load_session("session_after_mcmc4real_unsupervised_num_"+str(unsupervised_num)+"testset" + str(testset_num-1) + ".pkl")
        Z_points = np.arange(0, 1.21, 0.01)
        cum_cnt = 0
        for testset_ind, (_, test_index) in enumerate(kf.split(X[supervised_ind], input_label[supervised_ind])):
            dill.load_session("session_after_mcmc4real_unsupervised_num_"+str(unsupervised_num)+"testset" + str(testset_ind) + ".pkl")
            testsamp_num = len(Z_logprob_results[testset_ind])
            argmax_ind = np.argmax(Z_logprob_results[testset_ind],axis=1)
            predicted_Z[cum_cnt:cum_cnt + testsamp_num] = Z_points[argmax_ind]
            true_test_Z[cum_cnt:cum_cnt + testsamp_num] = supervised_test_Z[:,0]
            true_test_t[cum_cnt:cum_cnt + testsamp_num] = supervised_test_t
            cum_cnt += testsamp_num
            print(testset_ind)
            print(test_index)
        mse_z_pred = mean_squared_error(true_test_Z, predicted_Z)

        fig_truetestZ_vs_predictedZ = plt.figure(figsize=(5,5))
        plt.rcParams['font.size'] = '16'
        for t_val in range(4, 19, 1):
            plt.scatter(true_test_Z[true_test_t == t_val], predicted_Z[true_test_t == t_val],c='C0')
        plt.plot(np.array([-0.05,1.05]),np.array([-0.05,1.05]),'k--', linewidth=1)
        mse_text = 'Mean squared error: ' + '{:.6f}'.format(mse_z_pred)
        plt.text(0.0, 1.1, mse_text, fontsize=16)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('True abeta')
        plt.ylabel('Predicted abeta')
        plt.savefig("truetestZ_vs_predictedZ_unsupervised_num_"+str(unsupervised_num)+".eps")
        fig_truetestZ_vs_predictedZ.show()

        dill.dump_session("session_after_mcmc4real_unsupervised_num_" + str(unsupervised_num) + ".pkl")

    dill.dump_session("session_after_mcmc4real_final.pkl")