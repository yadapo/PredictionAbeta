#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Written by Yuichiro Yada
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

################################################
#　The notation is different from that in the paper.
#  paper,    this code
#  alpha,    beta
#  beta,     gamma
#  gamma,    q
#  z,        mu_Z
#  y,        Zo
################################################


import numpy as np

from jax.scipy.special import logsumexp

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from jax import random, vmap
import jax.numpy as jnp


def logistic(X, alpha, beta, q):
    tmp = -alpha * X - beta
    return q / (1. + np.exp(tmp))


def simulate_data(PRNGkey, sample_num, dim_X, dim_Z, dim_s, t=None, s=None, mu_alpha_prior=None, mu_beta_prior=None, mu_q_prior=None, mu_alpha_sigma=0.1, mu_beta_sigma=1, mu_q_sigma=0.1, prec_alpha_shape=100, prec_alpha_scale=1, prec_beta_shape=10, prec_beta_scale=1, prec_q_shape=25, prec_q_scale=0.1, W=None, sigma_W=None, sigma_X=None, grad_Z_correction_factor=1, wt_z_zero=0):
    PRNG_hpp, PRNG_s, PRNG_t, PRNG_alpha, PRNG_beta, PRNG_q, PRNG_Z, PRNG_sigX = random.split(PRNGkey, 8)
    sigma_Z = np.zeros([dim_Z])
    for Z_ind in range(dim_Z):
        prec_Z = numpyro.deterministic("prec_Z", jnp.array([400]))
        sigma_Z[Z_ind] = 1.0 / jnp.sqrt(prec_Z)

    mu_alpha = np.zeros([dim_s])
    sigma_alpha = np.zeros([dim_s])
    mu_beta = np.zeros([dim_s])
    sigma_beta = np.zeros([dim_s])
    mu_q = np.zeros([dim_s])
    sigma_q = np.zeros([dim_s])
    for s_ind in range(dim_s):
        mu_alpha[s_ind] = dist.Normal(loc=mu_alpha_prior[s_ind], scale=mu_alpha_sigma[s_ind]).sample(PRNG_hpp)
        prec_alpha = dist.Gamma(prec_alpha_shape[s_ind], prec_alpha_scale[s_ind]).sample(PRNG_hpp)
        sigma_alpha[s_ind] = 1.0/jnp.sqrt(prec_alpha)
        mu_beta[s_ind] = dist.Normal(loc=mu_beta_prior[s_ind], scale=mu_beta_sigma[s_ind]).sample(PRNG_hpp)
        prec_beta = dist.Gamma(prec_beta_shape[s_ind], prec_beta_scale[s_ind]).sample(PRNG_hpp)
        sigma_beta[s_ind] = 1.0/jnp.sqrt(prec_beta)
        mu_q[s_ind] = dist.Normal(loc=mu_q_prior[s_ind], scale=mu_q_sigma[s_ind]).sample(PRNG_hpp)
        prec_q = dist.Gamma(prec_q_shape[s_ind], prec_q_scale[s_ind]).sample(PRNG_hpp)
        sigma_q[s_ind] = 1.0 / jnp.sqrt(prec_q)

    if s is None:
        s = dist.Categorical(np.repeat(0.5, 2)).sample(PRNG_s, (sample_num,1))
        s = np.int8(np.squeeze(s))
    if t is None:
        t = dist.DiscreteUniform(4, 12).sample(PRNG_t, (sample_num,1))
        t = np.squeeze(t)
    mu_Z = np.zeros([sample_num,])
    grad_Z = np.zeros([sample_num,])
    alpha = dist.TruncatedNormal(low=0.0, loc=mu_alpha[s], scale=sigma_alpha[s]).sample(PRNG_alpha)
    beta = dist.Normal(loc=mu_beta[s], scale=sigma_beta[s]).sample(PRNG_beta)
    q = dist.TruncatedNormal(low=0.0, loc=mu_q[s], scale=sigma_q[s]).sample(PRNG_q)
    for sample_ind in range(sample_num):
        t_expanded = np.tile(t[sample_ind], (dim_Z,))
        mu_Z[sample_ind] = q[sample_ind] / (1 + jnp.exp(-jnp.transpose(alpha[sample_ind]) * t_expanded - jnp.transpose(beta[sample_ind])))
        grad_Z[sample_ind] = grad_Z_correction_factor * alpha[sample_ind] * mu_Z[sample_ind] * (1 - mu_Z[sample_ind] / q[sample_ind])

    Z = dist.TruncatedNormal(low=0.0, loc=mu_Z, scale=sigma_Z).sample(PRNG_Z)
    if wt_z_zero == 1:
        mu_Z[s == 1] = 0
        grad_Z[s == 1] = 0
        Z = Z.at[s == 1].set(0)

    Z = np.stack(Z)
    Zo = np.zeros([sample_num, 2*dim_Z+1])
    Zo[:, :dim_Z] = np.expand_dims(mu_Z,axis=1)
    Zo[:, dim_Z:2*dim_Z] = np.expand_dims(grad_Z,axis=1)
    Zo[:, 2*dim_Z] = np.ones(sample_num)

    if W is None:
        dim_Zo = Zo.shape[1]
        W = np.zeros([dim_X, dim_Zo])
        sigma_W = np.zeros([dim_X])
        W_gam_a = 100
        W_gam_b = 1000
        for row_ind in range(dim_X):
            prec = np.random.gamma(W_gam_a, 1 / W_gam_b)  # according to the implementation of gamma dist. in numpy.random
            sigma_W[row_ind] = 1 / np.sqrt(prec)
            W[row_ind] = np.random.multivariate_normal(np.repeat(0,dim_Zo), sigma_W[row_ind]*np.eye(dim_Zo))

    if sigma_X is None:
        prec_X = dist.Gamma(0.5, 1).sample(PRNG_sigX, sample_shape=(dim_X,))
        sigma_X = numpyro.deterministic("sigma_X", 1 / jnp.sqrt(prec_X))
    cov_mat_X = sigma_X ** 2 * jnp.eye(dim_X)

    X = np.zeros([sample_num, dim_X])
    mu_X = W @ np.transpose(Zo)
    for sample_ind in range(sample_num):
        X[sample_ind] = np.random.multivariate_normal(mu_X[:,sample_ind], cov_mat_X)

    return X, Z, t, s, alpha, beta, q, W, sigma_W, sigma_X, mu_alpha, sigma_alpha, mu_beta, sigma_beta, mu_q, sigma_q


def latent_model(Z_obs=None, t_obs=None, s_obs=None, mu_alpha_prior=None, mu_beta_prior=None, mu_q_prior=None, mu_alpha_sigma=0.1, mu_beta_sigma=1, mu_q_sigma=0.1, prec_alpha_shape=100, prec_alpha_scale=1, prec_beta_shape=10, prec_beta_scale=1, prec_q_shape=25, prec_q_scale=0.5):
    dim_s = 2
    sample_num = Z_obs.shape[0]
    dim_Z = 1

    # Hyper param.
    prec_Z = numpyro.deterministic("prec_Z", jnp.array([400]))
    sigma_Z = 1.0 / jnp.sqrt(prec_Z)
    with numpyro.plate("plate_type", dim_s):
        mu_alpha = numpyro.sample("mu_alpha", dist.Normal(loc=mu_alpha_prior, scale=mu_alpha_sigma))
        prec_alpha = numpyro.sample("prec_alpha", dist.Gamma(prec_alpha_shape, prec_alpha_scale))
        sigma_alpha = 1.0/jnp.sqrt(prec_alpha)
        mu_beta = numpyro.sample("mu_beta", dist.Normal(loc=mu_beta_prior, scale=mu_beta_sigma))
        prec_beta = numpyro.sample("prec_beta", dist.Gamma(prec_beta_shape, prec_beta_scale))
        sigma_beta = 1.0 / jnp.sqrt(prec_beta)
        mu_q = numpyro.sample("mu_q", dist.Normal(loc=mu_q_prior, scale=mu_q_sigma))
        prec_q = numpyro.sample("prec_q", dist.Gamma(prec_q_shape, prec_q_scale))
        sigma_q = 1.0 / jnp.sqrt(prec_q)

    with numpyro.plate("plate_sample", sample_num):
        s = numpyro.sample("s", dist.Categorical(jnp.repeat(0.5, 2)), obs=s_obs)
        t = numpyro.sample("t", dist.DiscreteUniform(2, 18), obs=t_obs)
        t_expanded = jnp.tile(t, (dim_Z, 1))
        alpha = numpyro.sample("alpha", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_alpha[s]), scale=jnp.transpose(sigma_alpha[s])))
        beta = numpyro.sample("beta", dist.Normal(loc=jnp.transpose(mu_beta[s]), scale=jnp.transpose(sigma_beta[s])))
        q = numpyro.sample("q", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_q[s]), scale=jnp.transpose(sigma_q[s])))
        mu_Z = q / (1 + jnp.exp(-alpha * t_expanded - beta))
        mu_Z_t0 = q / (1 + jnp.exp(-alpha * np.zeros([dim_Z, sample_num]) - beta))
        mu_Z = jnp.concatenate([mu_Z, mu_Z_t0])
        cov_mat_Z = (sigma_Z ** 2) * jnp.expand_dims(jnp.eye(dim_Z * 2), axis=2)
        numpyro.sample("Z", dist.MultivariateNormal(jnp.transpose(mu_Z), jnp.transpose(cov_mat_Z)), obs=np.concatenate([Z_obs, np.zeros([sample_num,dim_Z])],axis=1))


def full_model(X_obs=None, t_obs=None, s_obs=None, supervised_X_obs=None, supervised_Z_obs=None, supervised_t_obs=None, supervised_s_obs=None, mu_alpha_prior=None, mu_beta_prior=None, mu_q_prior=None, mu_alpha_sigma=0.1, mu_beta_sigma=0.1, mu_q_sigma=0.1, prec_alpha_shape=25, prec_alpha_scale=1, prec_beta_shape=25, prec_beta_scale=1, prec_q_shape=25, prec_q_scale=1, grad_Z_correction_factor=1):
    sample_num = X_obs.shape[0]
    supervised_sample_num = supervised_X_obs.shape[0]
    dim_X = X_obs.shape[1]
    dim_Z = 1
    dim_s = 2
    dim_Zo = 2*dim_Z + 1

    # Hyper param.
    with numpyro.plate("plate_latentdim", dim_Z):
        prec_Z = numpyro.deterministic("prec_Z", jnp.array([400]))
        sigma_Z = 1.0 / jnp.sqrt(prec_Z)

    with numpyro.plate("plate_type", dim_s):
        mu_alpha = numpyro.sample("mu_alpha", dist.Normal(loc=mu_alpha_prior, scale=mu_alpha_sigma))
        prec_alpha = numpyro.sample("prec_alpha", dist.Gamma(prec_alpha_shape, prec_alpha_scale))
        sigma_alpha = 1.0/jnp.sqrt(prec_alpha)
        mu_beta = numpyro.sample("mu_beta", dist.Normal(loc=mu_beta_prior, scale=mu_beta_sigma))
        prec_beta = numpyro.sample("prec_beta", dist.Gamma(prec_beta_shape, prec_beta_scale))
        sigma_beta = 1.0/jnp.sqrt(prec_beta)
        mu_q = numpyro.sample("mu_q", dist.Normal(loc=mu_q_prior, scale=mu_q_sigma))
        prec_q = numpyro.sample("prec_q", dist.Gamma(prec_q_shape, prec_q_scale))
        sigma_q = 1.0 / jnp.sqrt(prec_q)

    ################### for Gibbs #########################################
    with numpyro.plate("plate_xdim", dim_X):
        prec_W = numpyro.sample("prec_W", dist.Gamma(100, 1000))
        sigma_W = numpyro.deterministic("sigma_W", 1 / jnp.sqrt(prec_W))
        cov_mat_W = sigma_W ** 2 * jnp.transpose(jnp.tile(jnp.eye(dim_Zo), (dim_X, 1, 1)))
        W = numpyro.sample('W', dist.MultivariateNormal(jnp.repeat(0, dim_Zo), jnp.transpose(cov_mat_W)))
        prec_X = numpyro.sample("prec_X", dist.Gamma(0.5, 1))
        sigma_X = numpyro.deterministic("sigma_X", 1 / jnp.sqrt(prec_X))
        cov_mat_X = sigma_X ** 2 * jnp.eye(dim_X)
    #######################################################################

    with numpyro.plate("plate_supervised_sample", supervised_sample_num):
        s = numpyro.sample("supervised_s", dist.Categorical(jnp.repeat(0.5, 2)), obs=supervised_s_obs)
        t = numpyro.sample("supervised_t", dist.DiscreteUniform(2, 18), obs=supervised_t_obs)
        with numpyro.plate("plate_supervised_Z", dim_Z):
            alpha = numpyro.sample("alpha_supervised_samp", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_alpha[s]), scale=jnp.transpose(sigma_alpha[s])))
            beta = numpyro.sample("beta_supervised_samp", dist.Normal(loc=jnp.transpose(mu_beta[s]), scale=jnp.transpose(sigma_beta[s])))
            q = numpyro.sample("q_supervised_samp", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_q[s]), scale=jnp.transpose(sigma_q[s])))
        t_expanded = jnp.tile(t, (dim_Z, 1))
        mu_Z = q / (1 + jnp.exp(-alpha * t_expanded - beta))
        mu_Z = numpyro.deterministic("supervised_mu_Z", mu_Z)
        numpyro.sample("supervised_Z", dist.Normal(mu_Z, sigma_Z), obs=supervised_Z_obs)
        grad_Z = grad_Z_correction_factor * alpha * mu_Z * (1 - mu_Z/q)
        Zo = jnp.zeros([supervised_sample_num, dim_Zo])
        Zo = Zo.at[:, :dim_Z].set(jnp.transpose(mu_Z))
        Zo = Zo.at[:, dim_Z:2 * dim_Z].set(jnp.transpose(grad_Z))
        Zo = Zo.at[:, 2*dim_Z].set(jnp.ones(supervised_sample_num))
        Zo = numpyro.deterministic("Zo_supervised_samp", Zo)
        numpyro.sample("supervised_X", dist.MultivariateNormal(jnp.transpose(W @ jnp.transpose(Zo)), jnp.transpose(cov_mat_X)), obs=supervised_X_obs)

    with numpyro.plate("plate_sample", sample_num):
        s = numpyro.sample("s", dist.Categorical(jnp.repeat(0.5, 2)), obs=s_obs)
        t = numpyro.sample("t", dist.DiscreteUniform(2, 18), obs=t_obs)
        with numpyro.plate("plate_Z", dim_Z):
            alpha = numpyro.sample("alpha", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_alpha[s]), scale=jnp.transpose(sigma_alpha[s])))
            beta = numpyro.sample("beta", dist.Normal(loc=jnp.transpose(mu_beta[s]), scale=jnp.transpose(sigma_beta[s])))
            q = numpyro.sample("q", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_q[s]), scale=jnp.transpose(sigma_q[s])))
        t_expanded = jnp.tile(t, (dim_Z, 1))
        mu_Z = q / (1 + jnp.exp(-alpha * t_expanded - beta))
        Z = numpyro.deterministic("Z", mu_Z)
        grad_Z = grad_Z_correction_factor * alpha * Z * (1 - Z/q)
        Zo = jnp.zeros([sample_num, dim_Zo])
        Zo = Zo.at[:, :dim_Z].set(jnp.transpose(Z))
        Zo = Zo.at[:, dim_Z:2 * dim_Z].set(jnp.transpose(grad_Z))
        Zo = Zo.at[:, 2*dim_Z].set(jnp.ones(sample_num))
        Zo = numpyro.deterministic("Zo", Zo)
        numpyro.sample("X", dist.MultivariateNormal(jnp.transpose(W @ jnp.transpose(Zo)), jnp.transpose(cov_mat_X)), obs=X_obs)


def supervised_model(supervised_sample_num=None, supervised_X_obs=None, supervised_Z_obs=None, supervised_t_obs=None, supervised_s_obs=None, mu_alpha_prior=None, mu_beta_prior=None, mu_q_prior=None, mu_alpha_sigma=0.1, mu_beta_sigma=0.1, mu_q_sigma=0.1, prec_alpha_shape=25, prec_alpha_scale=1, prec_beta_shape=25, prec_beta_scale=1, prec_q_shape=25, prec_q_scale=1, grad_Z_correction_factor=1):
    dim_X = supervised_X_obs.shape[0]
    dim_Z = 1
    dim_s = 2
    dim_Zo = 2*dim_Z + 1

    # Hyper param.
    with numpyro.plate("plate_latentdim", dim_Z):
        prec_Z = numpyro.deterministic("prec_Z", jnp.array([400]))
        sigma_Z = 1.0 / jnp.sqrt(prec_Z)

    with numpyro.plate("plate_type", dim_s):
        mu_alpha = numpyro.sample("mu_alpha", dist.Normal(loc=mu_alpha_prior, scale=mu_alpha_sigma))
        prec_alpha = numpyro.sample("prec_alpha", dist.Gamma(prec_alpha_shape, prec_alpha_scale))
        sigma_alpha = 1.0/jnp.sqrt(prec_alpha)
        mu_beta = numpyro.sample("mu_beta", dist.Normal(loc=mu_beta_prior, scale=mu_beta_sigma))
        prec_beta = numpyro.sample("prec_beta", dist.Gamma(prec_beta_shape, prec_beta_scale))
        sigma_beta = 1.0/jnp.sqrt(prec_beta)
        mu_q = numpyro.sample("mu_q", dist.Normal(loc=mu_q_prior, scale=mu_q_sigma))
        prec_q = numpyro.sample("prec_q", dist.Gamma(prec_q_shape, prec_q_scale))
        sigma_q = 1.0 / jnp.sqrt(prec_q)

    ################### for Gibbs #########################################
    with numpyro.plate("plate_xdim", dim_X):
        prec_W = numpyro.sample("prec_W", dist.Gamma(100, 1000))
        sigma_W = numpyro.deterministic("sigma_W", 1 / jnp.sqrt(prec_W))
        cov_mat_W = sigma_W ** 2 * jnp.transpose(jnp.tile(jnp.eye(dim_Zo), (dim_X, 1, 1)))
        W = numpyro.sample('W', dist.MultivariateNormal(jnp.repeat(0, dim_Zo), jnp.transpose(cov_mat_W)))
        prec_X = numpyro.sample("prec_X", dist.Gamma(0.5, 1))
        sigma_X = numpyro.deterministic("sigma_X", 1 / jnp.sqrt(prec_X))
        cov_mat_X = sigma_X ** 2 * jnp.eye(dim_X)
    #######################################################################

    with numpyro.plate("plate_supervised_sample", supervised_sample_num):
        s = numpyro.sample("supervised_s", dist.Categorical(jnp.repeat(0.5, 2)), obs=supervised_s_obs)
        t = numpyro.sample("supervised_t", dist.DiscreteUniform(2, 18), obs=supervised_t_obs)
        with numpyro.plate("plate_supervised_Z", dim_Z):
            alpha = numpyro.sample("alpha_supervised_samp", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_alpha[s]), scale=jnp.transpose(sigma_alpha[s])))
            beta = numpyro.sample("beta_supervised_samp", dist.Normal(loc=jnp.transpose(mu_beta[s]), scale=jnp.transpose(sigma_beta[s])))
            q = numpyro.sample("q_supervised_samp", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_q[s]), scale=jnp.transpose(sigma_q[s])))
        t_expanded = jnp.tile(t, (dim_Z, 1))
        mu_Z = q / (1 + jnp.exp(-alpha * t_expanded - beta))
        mu_Z = numpyro.deterministic("supervised_mu_Z", mu_Z)
        numpyro.sample("supervised_Z", dist.Normal(mu_Z, sigma_Z), obs=supervised_Z_obs)
        grad_Z = grad_Z_correction_factor * alpha * mu_Z * (1 - mu_Z/q)
        Zo = jnp.zeros([supervised_sample_num, dim_Zo])
        Zo = Zo.at[:, :dim_Z].set(jnp.transpose(mu_Z))
        Zo = Zo.at[:, dim_Z:2 * dim_Z].set(jnp.transpose(grad_Z))
        Zo = Zo.at[:, 2*dim_Z].set(jnp.ones(supervised_sample_num))
        numpyro.sample("supervised_X", dist.MultivariateNormal(jnp.transpose(W @ jnp.transpose(Zo)), jnp.transpose(cov_mat_X)), obs=supervised_X_obs)


def unsupervised_model(X_obs=None, t_obs=None, s_obs=None, mu_alpha_prior=None, mu_beta_prior=None, mu_q_prior=None, mu_alpha_sigma=0.1, mu_beta_sigma=0.1, mu_q_sigma=0.1, prec_alpha_shape=25, prec_alpha_scale=1, prec_beta_shape=25, prec_beta_scale=1, prec_q_shape=25, prec_q_scale=1, grad_Z_correction_factor=1):
    sample_num = X_obs.shape[0]
    dim_X = X_obs.shape[1]
    dim_Z = 1
    dim_s = 2
    dim_Zo = 2*dim_Z + 1

    # Hyper param.
    with numpyro.plate("plate_latentdim", dim_Z):
        prec_Z = numpyro.deterministic("prec_Z", jnp.array([400]))
        sigma_Z = 1.0 / jnp.sqrt(prec_Z)

    with numpyro.plate("plate_type", dim_s):
        mu_alpha = numpyro.sample("mu_alpha", dist.Normal(loc=mu_alpha_prior, scale=mu_alpha_sigma))
        prec_alpha = numpyro.sample("prec_alpha", dist.Gamma(prec_alpha_shape, prec_alpha_scale))
        sigma_alpha = 1.0/jnp.sqrt(prec_alpha)
        mu_beta = numpyro.sample("mu_beta", dist.Normal(loc=mu_beta_prior, scale=mu_beta_sigma))
        prec_beta = numpyro.sample("prec_beta", dist.Gamma(prec_beta_shape, prec_beta_scale))
        sigma_beta = 1.0/jnp.sqrt(prec_beta)
        mu_q = numpyro.sample("mu_q", dist.Normal(loc=mu_q_prior, scale=mu_q_sigma))
        prec_q = numpyro.sample("prec_q", dist.Gamma(prec_q_shape, prec_q_scale))
        sigma_q = 1.0 / jnp.sqrt(prec_q)

    ################### for Gibbs #########################################
    with numpyro.plate("plate_xdim", dim_X):
        prec_W = numpyro.sample("prec_W", dist.Gamma(100, 1000))
        sigma_W = numpyro.deterministic("sigma_W", 1 / jnp.sqrt(prec_W))
        cov_mat_W = sigma_W ** 2 * jnp.transpose(jnp.tile(jnp.eye(dim_Zo), (dim_X, 1, 1)))
        W = numpyro.sample('W', dist.MultivariateNormal(jnp.repeat(0, dim_Zo), jnp.transpose(cov_mat_W)))
        prec_X = numpyro.sample("prec_X", dist.Gamma(0.5, 1))
        sigma_X = numpyro.deterministic("sigma_X", 1 / jnp.sqrt(prec_X))
        cov_mat_X = sigma_X ** 2 * jnp.eye(dim_X)
    #######################################################################

    with numpyro.plate("plate_sample", sample_num):
        s = numpyro.sample("s", dist.Categorical(jnp.repeat(0.5, 2)), obs=s_obs)
        t = numpyro.sample("t", dist.DiscreteUniform(2, 18), obs=t_obs)
        with numpyro.plate("plate_Z", dim_Z):
            alpha = numpyro.sample("alpha", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_alpha[s]), scale=jnp.transpose(sigma_alpha[s])))
            beta = numpyro.sample("beta", dist.Normal(loc=jnp.transpose(mu_beta[s]), scale=jnp.transpose(sigma_beta[s])))
            q = numpyro.sample("q", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_q[s]), scale=jnp.transpose(sigma_q[s])))
        t_expanded = jnp.tile(t, (dim_Z, 1))
        mu_Z = q / (1 + jnp.exp(-alpha * t_expanded - beta))
        Z = numpyro.deterministic("Z", mu_Z)
        grad_Z = grad_Z_correction_factor * alpha * Z * (1 - Z/q)
        Zo = jnp.zeros([sample_num, dim_Zo])
        Zo = Zo.at[:, :dim_Z].set(jnp.transpose(Z))
        Zo = Zo.at[:, dim_Z:2 * dim_Z].set(jnp.transpose(grad_Z))
        Zo = Zo.at[:, 2*dim_Z].set(jnp.ones(sample_num))
        Zo = numpyro.deterministic("Zo", Zo)
        numpyro.sample("X", dist.MultivariateNormal(jnp.transpose(W @ jnp.transpose(Zo)), jnp.transpose(cov_mat_X)), obs=X_obs)


def log_likelihood_Z(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace['supervised_Z']
    Z_logprob = obs_node['fn'].log_prob(obs_node['value'])
    obs_node = model_trace['supervised_X']
    X_logprob = obs_node['fn'].log_prob(obs_node['value'])
    return Z_logprob + X_logprob


def log_likelihood_Z_given_ts(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace['supervised_t']
    t_logprob = obs_node['fn'].log_prob(obs_node['value'])
    obs_node = model_trace['supervised_s']
    s_logprob = obs_node['fn'].log_prob(obs_node['value'])
    obs_node = model_trace['supervised_Z']
    Z_logprob = obs_node['fn'].log_prob(obs_node['value'])
    obs_node = model_trace['supervised_X']
    X_logprob = obs_node['fn'].log_prob(obs_node['value'])
    return Z_logprob + X_logprob


def log_pred_density_Z(rng_key, params, model, *args, **kwargs):
    n = list(params.values())[0].shape[0]
    log_lk_fn = vmap(
        lambda rng_key, params: log_likelihood_Z(rng_key, params, model, *args, **kwargs)
    )
    log_lk_vals = log_lk_fn(random.split(rng_key, n), params)
    return (logsumexp(log_lk_vals, 0) - jnp.log(n)).sum()


def log_pred_density_Z_given_ts(rng_key, params, model, *args, **kwargs):
    n = list(params.values())[0].shape[0]
    log_lk_fn = vmap(
        lambda rng_key, params: log_likelihood_Z_given_ts(rng_key, params, model, *args, **kwargs)
    )
    log_lk_vals = log_lk_fn(random.split(rng_key, n), params)
    return (logsumexp(log_lk_vals, 0) - jnp.log(n)).sum()


def predictive_Z(new_X_obs=None, new_Z_obs=None, new_t_obs=None, new_s_obs=None, grad_Z_correction_factor=1, posterior_samples=None):
    keys = ['prec_X', 'W', 'prec_W', 'mu_alpha', 'prec_alpha', 'mu_beta', 'prec_beta', 'mu_q', 'prec_q']
    use_posterior_samples = {key:posterior_samples.get(key) for key in keys}

    rng_key = random.PRNGKey(111)
    if new_t_obs is not None:
        if new_s_obs is not None:
            log_pred = log_pred_density_Z(rng_key, use_posterior_samples, supervised_model, supervised_sample_num=1,
                                      supervised_X_obs=new_X_obs, supervised_Z_obs=new_Z_obs, supervised_t_obs=new_t_obs, supervised_s_obs=new_s_obs,
                                      grad_Z_correction_factor=grad_Z_correction_factor)
        else:
            log_pred = log_pred_density_Z(rng_key, use_posterior_samples, supervised_model, supervised_sample_num=1,
                                      supervised_X_obs=new_X_obs, supervised_Z_obs=new_Z_obs, supervised_t_obs=new_t_obs,
                                      grad_Z_correction_factor=grad_Z_correction_factor)
    elif new_s_obs is not None:
        log_pred = log_pred_density_Z(rng_key, use_posterior_samples, supervised_model, supervised_sample_num=1,
                                      supervised_X_obs=new_X_obs, supervised_Z_obs=new_Z_obs, supervised_s_obs=new_s_obs,
                                      grad_Z_correction_factor=grad_Z_correction_factor)
    else:
        log_pred = log_pred_density_Z(rng_key, use_posterior_samples, supervised_model, supervised_sample_num=1, supervised_X_obs=new_X_obs, supervised_Z_obs=new_Z_obs, grad_Z_correction_factor=grad_Z_correction_factor)
    return log_pred


def log_likelihood_s(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace['supervised_s']
    s_logprob = obs_node['fn'].log_prob(obs_node['value'])
    obs_node = model_trace['supervised_X']
    X_logprob = obs_node['fn'].log_prob(obs_node['value'])
    return X_logprob + s_logprob


def log_pred_density_s(rng_key, params, model, *args, **kwargs):
    n = list(params.values())[0].shape[0]
    log_lk_fn = vmap(
        lambda rng_key, params: log_likelihood_s(rng_key, params, model, *args, **kwargs)
    )
    log_lk_vals = log_lk_fn(random.split(rng_key, n), params)
    return (logsumexp(log_lk_vals, 0) - jnp.log(n)).sum()


def predictive_s(new_X_obs=None, new_s_obs=None, posterior_samples=None, grad_Z_correction_factor=1):
    keys = ['prec_X', 'W', 'prec_W', 'mu_alpha', 'prec_alpha', 'mu_beta', 'prec_beta', 'mu_q', 'prec_q']
    use_posterior_samples = {key:posterior_samples.get(key) for key in keys}

    rng_key = random.PRNGKey(111)
    log_pred = log_pred_density_s(rng_key, use_posterior_samples, supervised_model, supervised_sample_num = 1, supervised_X_obs=new_X_obs, supervised_s_obs=new_s_obs, grad_Z_correction_factor=grad_Z_correction_factor)

    return log_pred


def _gibbs_fn_full(X, rng_key, gibbs_sites, hmc_sites):
    PRNGkey = rng_key
    Zo_unsupervised = hmc_sites["Zo"]
    Zo_supervised = hmc_sites["Zo_supervised_samp"]
    sigma_W = gibbs_sites["sigma_W"]
    sigma_X = gibbs_sites["sigma_X"]

    Zo = jnp.concatenate([Zo_unsupervised,Zo_supervised])

    dim_X = X.shape[1]
    dim_Z = 1

    #####
    # sampling W
    W = sample_W(PRNGkey, X, sigma_X, Zo, sigma_W)

    #####
    # sampling sigma W
    sigma_W = sample_sigma_W(PRNGkey, W)

    #####
    # sampling sigma X
    X_gam_a = jnp.ones([dim_X, ]) * 0.5
    X_gam_b = jnp.ones([dim_X, ]) * 1
    sigma_X = sample_sigma_X(PRNGkey, X, Zo, W, X_gam_a, X_gam_b)

    return{"W":W, "sigma_W":sigma_W, "sigma_X":sigma_X}


def _gibbs_fn_unsupervised(X, rng_key, gibbs_sites, hmc_sites):
    PRNGkey = rng_key
    Zo = hmc_sites["Zo"]
    sigma_W = gibbs_sites["sigma_W"]
    sigma_X = gibbs_sites["sigma_X"]

    dim_X = X.shape[1]
    dim_Z = 1

    #####
    # sampling W
    W = sample_W(PRNGkey, X, sigma_X, Zo, sigma_W)

    #####
    # sampling sigma W
    sigma_W = sample_sigma_W(PRNGkey, W)

    #####
    # sampling sigma X
    X_gam_a = jnp.ones([dim_X, ]) * 0.5
    X_gam_b = jnp.ones([dim_X, ]) * 1
    sigma_X = sample_sigma_X(PRNGkey, X, Zo, W, X_gam_a, X_gam_b)

    return{"W":W, "sigma_W":sigma_W, "sigma_X":sigma_X}


def sample_W(PRNGkey, X, sigma_X, Zo, sigma_W):
    sample_num = X.shape[0]
    dim_X = X.shape[1]
    dim_Zo = Zo.shape[1]
    W = jnp.zeros([dim_X, dim_Zo])
    cov_mat_W = jnp.transpose(sigma_W ** 2 * jnp.transpose(jnp.tile(jnp.eye(dim_Zo), (dim_X, 1, 1))))
    post_mu_W = jnp.zeros([dim_X, dim_Zo])
    post_cov_mat_W = jnp.zeros([dim_X, dim_Zo, dim_Zo])
    for row_ind in range(dim_X):
        tmp = jnp.zeros([dim_Zo, dim_Zo])
        for sample_ind in range(sample_num):
            tmp += jnp.outer(Zo[sample_ind], Zo[sample_ind])
        inv_post_cov_mat_W_row = tmp / sigma_X[row_ind] ** 2 + jnp.linalg.inv(cov_mat_W[row_ind])
        post_cov_mat_W = post_cov_mat_W.at[row_ind].set(jnp.linalg.inv(inv_post_cov_mat_W_row))
        post_mu_W = post_mu_W.at[row_ind].set(post_cov_mat_W[row_ind] @ (jnp.sum(np.transpose(Zo) * jnp.transpose(X[:, row_ind]) / sigma_X[row_ind]**2, axis=1)))
        W = W.at[row_ind].set(dist.MultivariateNormal(post_mu_W[row_ind], post_cov_mat_W[row_ind]).sample(PRNGkey))
    return W


def sample_sigma_X(PRNGkey, X, Zo, W, X_gam_a, X_gam_b):
    sample_num = X.shape[0]
    dim_X = X.shape[1]
    sigma_X = jnp.zeros([dim_X])
    post_X_gam_a = jnp.zeros([dim_X])
    post_X_gam_b = jnp.zeros([dim_X])
    for row_ind in range(dim_X):
        post_X_gam_a = post_X_gam_a.at[row_ind].set(X_gam_a[row_ind] + sample_num/2)
        post_X_gam_b = post_X_gam_b.at[row_ind].set(jnp.sum(((X[:, row_ind] - W[row_ind] @ jnp.transpose(Zo))**2)/2) + X_gam_b[row_ind])
        prec = dist.Gamma(post_X_gam_a[row_ind], 1/post_X_gam_b[row_ind]).sample(PRNGkey)
        sigma_X = sigma_X.at[row_ind].set(1 / jnp.sqrt(prec))
    return sigma_X


def sample_sigma_W(PRNGkey, W):
    dim_X = W.shape[0]
    dim_Zo = W.shape[1]
    sigma_W = jnp.zeros([dim_X])
    a = 100
    b = 1000
    post_W_gam_a = dim_Zo * 1/2 + a
    for row_ind in range(dim_X):
        post_W_gam_b = W[row_ind] @ W[row_ind] / 2 + b
        prec = dist.Gamma(post_W_gam_a, post_W_gam_b).sample(PRNGkey)
        sigma_W = sigma_W.at[row_ind].set(1 / jnp.sqrt(prec))
    return sigma_W


################################
# Neural network
#####################
def nn_full_model(X_obs=None, t_obs=None, s_obs=None, supervised_X_obs=None, supervised_Z_obs=None, supervised_t_obs=None, supervised_s_obs=None, mu_alpha_prior=None, mu_beta_prior=None, mu_q_prior=None, mu_alpha_sigma=0.1, mu_beta_sigma=0.1, mu_q_sigma=0.1, prec_alpha_shape=25, prec_alpha_scale=1, prec_beta_shape=25, prec_beta_scale=1, prec_q_shape=25, prec_q_scale=1, grad_Z_correction_factor=1):
    sample_num = X_obs.shape[0]
    supervised_sample_num = supervised_X_obs.shape[0]
    dim_X = X_obs.shape[1]
    dim_Z = 1
    dim_s = 2
    dim_Zo = 2*dim_Z + 1

    # Hyper param.
    with numpyro.plate("plate_latentdim", dim_Z):
        prec_Z = numpyro.deterministic("prec_Z", jnp.array([400]))
        sigma_Z = 1.0 / jnp.sqrt(prec_Z)

    with numpyro.plate("plate_type", dim_s):
        mu_alpha = numpyro.sample("mu_alpha", dist.Normal(loc=mu_alpha_prior, scale=mu_alpha_sigma))
        prec_alpha = numpyro.sample("prec_alpha", dist.Gamma(prec_alpha_shape, prec_alpha_scale))
        sigma_alpha = 1.0/jnp.sqrt(prec_alpha)
        mu_beta = numpyro.sample("mu_beta", dist.Normal(loc=mu_beta_prior, scale=mu_beta_sigma))
        prec_beta = numpyro.sample("prec_beta", dist.Gamma(prec_beta_shape, prec_beta_scale))
        sigma_beta = 1.0/jnp.sqrt(prec_beta)
        mu_q = numpyro.sample("mu_q", dist.Normal(loc=mu_q_prior, scale=mu_q_sigma))
        prec_q = numpyro.sample("prec_q", dist.Gamma(prec_q_shape, prec_q_scale))
        sigma_q = 1.0 / jnp.sqrt(prec_q)

    with numpyro.plate("plate_xdim", dim_X):
        prec_X = numpyro.sample("prec_X", dist.Gamma(0.05, 1))
        sigma_X = numpyro.deterministic("sigma_X", 1 / jnp.sqrt(prec_X))
        cov_mat_X = sigma_X ** 2 * jnp.eye(dim_X)

    V1_dim = 3
    V2_dim = 11
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((dim_Zo, V1_dim)), jnp.ones((dim_Zo, V1_dim))))
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((V1_dim, V2_dim)), jnp.ones((V1_dim, V2_dim))))
    W3 = numpyro.sample("W3", dist.Normal(jnp.zeros((V2_dim, dim_X)), jnp.ones((V2_dim, dim_X))))

    with numpyro.plate("plate_supervised_sample", supervised_sample_num):
        s = numpyro.sample("supervised_s", dist.Categorical(jnp.repeat(0.5, 2)), obs=supervised_s_obs)
        t = numpyro.sample("supervised_t", dist.DiscreteUniform(2, 18), obs=supervised_t_obs)
        with numpyro.plate("plate_supervised_Z", dim_Z):
            alpha = numpyro.sample("alpha_supervised_samp", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_alpha[s]), scale=jnp.transpose(sigma_alpha[s])))
            beta = numpyro.sample("beta_supervised_samp", dist.Normal(loc=jnp.transpose(mu_beta[s]), scale=jnp.transpose(sigma_beta[s])))
            q = numpyro.sample("q_supervised_samp", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_q[s]), scale=jnp.transpose(sigma_q[s])))
        t_expanded = jnp.tile(t, (dim_Z, 1))
        mu_Z = q / (1 + jnp.exp(-alpha * t_expanded - beta))
        mu_Z = numpyro.deterministic("supervised_mu_Z", mu_Z)
        numpyro.sample("supervised_Z", dist.Normal(mu_Z, sigma_Z), obs=supervised_Z_obs)
        grad_Z = grad_Z_correction_factor * alpha * mu_Z * (1 - mu_Z/q)
        Zo = jnp.zeros([supervised_sample_num, dim_Zo])
        Zo = Zo.at[:, :dim_Z].set(jnp.transpose(mu_Z))
        Zo = Zo.at[:, dim_Z:2 * dim_Z].set(jnp.transpose(grad_Z))
        Zo = Zo.at[:, 2*dim_Z].set(jnp.ones(supervised_sample_num))
        Zo = numpyro.deterministic("Zo_supervised_samp", Zo)
        V1 = jnp.tanh(jnp.matmul(Zo, W1))
        V2 = jnp.tanh(jnp.matmul(V1, W2))
        V3 = jnp.matmul(V2, W3)
        numpyro.sample("supervised_X", dist.MultivariateNormal(V3, jnp.transpose(cov_mat_X)), obs=supervised_X_obs)

    with numpyro.plate("plate_sample", sample_num):
        s = numpyro.sample("s", dist.Categorical(jnp.repeat(0.5, 2)), obs=s_obs)
        t = numpyro.sample("t", dist.DiscreteUniform(2, 18), obs=t_obs)
        with numpyro.plate("plate_Z", dim_Z):
            alpha = numpyro.sample("alpha", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_alpha[s]), scale=jnp.transpose(sigma_alpha[s])))
            beta = numpyro.sample("beta", dist.Normal(loc=jnp.transpose(mu_beta[s]), scale=jnp.transpose(sigma_beta[s])))
            q = numpyro.sample("q", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_q[s]), scale=jnp.transpose(sigma_q[s])))
        t_expanded = jnp.tile(t, (dim_Z, 1))
        mu_Z = q / (1 + jnp.exp(-alpha * t_expanded - beta))
        Z = numpyro.deterministic("Z", mu_Z)
        grad_Z = grad_Z_correction_factor * alpha * Z * (1 - Z/q)
        Zo = jnp.zeros([sample_num, dim_Zo])
        Zo = Zo.at[:, :dim_Z].set(jnp.transpose(Z))
        Zo = Zo.at[:, dim_Z:2 * dim_Z].set(jnp.transpose(grad_Z))
        Zo = Zo.at[:, 2*dim_Z].set(jnp.ones(sample_num))
        Zo = numpyro.deterministic("Zo", Zo)
        V1 = jnp.tanh(jnp.matmul(Zo, W1))
        V2 = jnp.tanh(jnp.matmul(V1, W2))
        V3 = jnp.matmul(V2, W3)
        numpyro.sample("X", dist.MultivariateNormal(V3, jnp.transpose(cov_mat_X)), obs=X_obs)


def nn_supervised_model(supervised_sample_num=None, supervised_X_obs=None, supervised_Z_obs=None, supervised_t_obs=None, supervised_s_obs=None, mu_alpha_prior=None, mu_beta_prior=None, mu_q_prior=None, mu_alpha_sigma=0.1, mu_beta_sigma=0.1, mu_q_sigma=0.1, prec_alpha_shape=25, prec_alpha_scale=1, prec_beta_shape=25, prec_beta_scale=1, prec_q_shape=25, prec_q_scale=1, grad_Z_correction_factor=1):
    dim_X = supervised_X_obs.shape[0]
    dim_Z = 1
    dim_s = 2
    dim_Zo = 2*dim_Z + 1

    # Hyper param.
    with numpyro.plate("plate_latentdim", dim_Z):
        prec_Z = numpyro.deterministic("prec_Z", jnp.array([400]))
        sigma_Z = 1.0 / jnp.sqrt(prec_Z)

    with numpyro.plate("plate_type", dim_s):
        mu_alpha = numpyro.sample("mu_alpha", dist.Normal(loc=mu_alpha_prior, scale=mu_alpha_sigma))
        prec_alpha = numpyro.sample("prec_alpha", dist.Gamma(prec_alpha_shape, prec_alpha_scale))
        sigma_alpha = 1.0/jnp.sqrt(prec_alpha)
        mu_beta = numpyro.sample("mu_beta", dist.Normal(loc=mu_beta_prior, scale=mu_beta_sigma))
        prec_beta = numpyro.sample("prec_beta", dist.Gamma(prec_beta_shape, prec_beta_scale))
        sigma_beta = 1.0/jnp.sqrt(prec_beta)
        mu_q = numpyro.sample("mu_q", dist.Normal(loc=mu_q_prior, scale=mu_q_sigma))
        prec_q = numpyro.sample("prec_q", dist.Gamma(prec_q_shape, prec_q_scale))
        sigma_q = 1.0 / jnp.sqrt(prec_q)

    with numpyro.plate("plate_xdim", dim_X):
        prec_X = numpyro.sample("prec_X", dist.Gamma(0.05, 1))
        sigma_X = numpyro.deterministic("sigma_X", 1 / jnp.sqrt(prec_X))
        cov_mat_X = sigma_X ** 2 * jnp.eye(dim_X)

    V1_dim = 3
    V2_dim = 11
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((dim_Zo, V1_dim)), jnp.ones((dim_Zo, V1_dim))))
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((V1_dim, V2_dim)), jnp.ones((V1_dim, V2_dim))))
    W3 = numpyro.sample("W3", dist.Normal(jnp.zeros((V2_dim, dim_X)), jnp.ones((V2_dim, dim_X))))

    with numpyro.plate("plate_supervised_sample", supervised_sample_num):
        s = numpyro.sample("supervised_s", dist.Categorical(jnp.repeat(0.5, 2)), obs=supervised_s_obs)
        t = numpyro.sample("supervised_t", dist.DiscreteUniform(2, 18), obs=supervised_t_obs)
        with numpyro.plate("plate_supervised_Z", dim_Z):
            alpha = numpyro.sample("alpha_supervised_samp", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_alpha[s]), scale=jnp.transpose(sigma_alpha[s])))
            beta = numpyro.sample("beta_supervised_samp", dist.Normal(loc=jnp.transpose(mu_beta[s]), scale=jnp.transpose(sigma_beta[s])))
            q = numpyro.sample("q_supervised_samp", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_q[s]), scale=jnp.transpose(sigma_q[s])))
        t_expanded = jnp.tile(t, (dim_Z, 1))
        mu_Z = q / (1 + jnp.exp(-alpha * t_expanded - beta))
        mu_Z = numpyro.deterministic("supervised_mu_Z", mu_Z)
        numpyro.sample("supervised_Z", dist.Normal(mu_Z, sigma_Z), obs=supervised_Z_obs)
        grad_Z = grad_Z_correction_factor * alpha * mu_Z * (1 - mu_Z/q)
        Zo = jnp.zeros([supervised_sample_num, dim_Zo])
        Zo = Zo.at[:, :dim_Z].set(jnp.transpose(mu_Z))
        Zo = Zo.at[:, dim_Z:2 * dim_Z].set(jnp.transpose(grad_Z))
        Zo = Zo.at[:, 2*dim_Z].set(jnp.ones(supervised_sample_num))
        V1 = jnp.tanh(jnp.matmul(Zo, W1))
        V2 = jnp.tanh(jnp.matmul(V1, W2))
        V3 = jnp.matmul(V2, W3)
        numpyro.sample("supervised_X", dist.MultivariateNormal(V3, jnp.transpose(cov_mat_X)), obs=supervised_X_obs)


def nn_unsupervised_model(X_obs=None, t_obs=None, s_obs=None, mu_alpha_prior=None, mu_beta_prior=None, mu_q_prior=None, mu_alpha_sigma=0.1, mu_beta_sigma=0.1, mu_q_sigma=0.1, prec_alpha_shape=25, prec_alpha_scale=1, prec_beta_shape=25, prec_beta_scale=1, prec_q_shape=25, prec_q_scale=1, grad_Z_correction_factor=1):
    sample_num = X_obs.shape[0]
    dim_X = X_obs.shape[1]
    dim_Z = 1
    dim_s = 2
    dim_Zo = 2*dim_Z + 1

    # Hyper param.
    with numpyro.plate("plate_latentdim", dim_Z):
        prec_Z = numpyro.deterministic("prec_Z", jnp.array([400]))

    with numpyro.plate("plate_type", dim_s):
        mu_alpha = numpyro.sample("mu_alpha", dist.Normal(loc=mu_alpha_prior, scale=mu_alpha_sigma))
        prec_alpha = numpyro.sample("prec_alpha", dist.Gamma(prec_alpha_shape, prec_alpha_scale))
        sigma_alpha = 1.0/jnp.sqrt(prec_alpha)
        mu_beta = numpyro.sample("mu_beta", dist.Normal(loc=mu_beta_prior, scale=mu_beta_sigma))
        prec_beta = numpyro.sample("prec_beta", dist.Gamma(prec_beta_shape, prec_beta_scale))
        sigma_beta = 1.0/jnp.sqrt(prec_beta)
        mu_q = numpyro.sample("mu_q", dist.Normal(loc=mu_q_prior, scale=mu_q_sigma))
        prec_q = numpyro.sample("prec_q", dist.Gamma(prec_q_shape, prec_q_scale))
        sigma_q = 1.0 / jnp.sqrt(prec_q)

    with numpyro.plate("plate_xdim", dim_X):
        prec_X = numpyro.sample("prec_X", dist.Gamma(0.05, 1))
        sigma_X = numpyro.deterministic("sigma_X", 1 / jnp.sqrt(prec_X))
        cov_mat_X = sigma_X ** 2 * jnp.eye(dim_X)

    V1_dim = 3
    V2_dim = 11
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((dim_Zo, V1_dim)), jnp.ones((dim_Zo, V1_dim))))
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((V1_dim, V2_dim)), jnp.ones((V1_dim, V2_dim))))
    W3 = numpyro.sample("W3", dist.Normal(jnp.zeros((V2_dim, dim_X)), jnp.ones((V2_dim, dim_X))))

    with numpyro.plate("plate_sample", sample_num):
        s = numpyro.sample("s", dist.Categorical(jnp.repeat(0.5, 2)), obs=s_obs)
        t = numpyro.sample("t", dist.DiscreteUniform(2, 18), obs=t_obs)
        with numpyro.plate("plate_Z", dim_Z):
            alpha = numpyro.sample("alpha", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_alpha[s]), scale=jnp.transpose(sigma_alpha[s])))
            beta = numpyro.sample("beta", dist.Normal(loc=jnp.transpose(mu_beta[s]), scale=jnp.transpose(sigma_beta[s])))
            q = numpyro.sample("q", dist.TruncatedNormal(low=0.0, loc=jnp.transpose(mu_q[s]), scale=jnp.transpose(sigma_q[s])))
        t_expanded = jnp.tile(t, (dim_Z, 1))
        mu_Z = q / (1 + jnp.exp(-alpha * t_expanded - beta))
        Z = numpyro.deterministic("Z", mu_Z)
        grad_Z = grad_Z_correction_factor * alpha * Z * (1 - Z/q)
        Zo = jnp.zeros([sample_num, dim_Zo])
        Zo = Zo.at[:, :dim_Z].set(jnp.transpose(Z))
        Zo = Zo.at[:, dim_Z:2 * dim_Z].set(jnp.transpose(grad_Z))
        Zo = Zo.at[:, 2*dim_Z].set(jnp.ones(sample_num))
        Zo = numpyro.deterministic("Zo", Zo)
        V1 = jnp.tanh(jnp.matmul(Zo, W1))
        V2 = jnp.tanh(jnp.matmul(V1, W2))
        V3 = jnp.matmul(V2, W3)
        numpyro.sample("X", dist.MultivariateNormal(V3, jnp.transpose(cov_mat_X)), obs=X_obs)


def nn_predictive_Z(new_X_obs=None, new_Z_obs=None, new_t_obs=None, new_s_obs=None, grad_Z_correction_factor=1, posterior_samples=None):
    keys = ['prec_X', 'W1', 'W2', 'W3', 'mu_alpha', 'prec_alpha', 'mu_beta', 'prec_beta', 'mu_q', 'prec_q']
    use_posterior_samples = {key:posterior_samples.get(key) for key in keys}

    rng_key = random.PRNGKey(111)
    if new_t_obs is not None:
        if new_s_obs is not None:
            log_pred = log_pred_density_Z(rng_key, use_posterior_samples, nn_supervised_model, supervised_sample_num=1,
                                      supervised_X_obs=new_X_obs, supervised_Z_obs=new_Z_obs, supervised_t_obs=new_t_obs, supervised_s_obs=new_s_obs,
                                      grad_Z_correction_factor=grad_Z_correction_factor)
        else:
            log_pred = log_pred_density_Z(rng_key, use_posterior_samples, nn_supervised_model, supervised_sample_num=1,
                                          supervised_X_obs=new_X_obs, supervised_Z_obs=new_Z_obs,
                                          supervised_t_obs=new_t_obs,
                                          grad_Z_correction_factor=grad_Z_correction_factor)
    elif new_s_obs is not None:
        log_pred = log_pred_density_Z(rng_key, use_posterior_samples, nn_supervised_model, supervised_sample_num=1,
                                      supervised_X_obs=new_X_obs, supervised_Z_obs=new_Z_obs, supervised_s_obs=new_s_obs,
                                      grad_Z_correction_factor=grad_Z_correction_factor)
    else:
        log_pred = log_pred_density_Z(rng_key, use_posterior_samples, nn_supervised_model, supervised_sample_num=1, supervised_X_obs=new_X_obs, supervised_Z_obs=new_Z_obs, grad_Z_correction_factor=grad_Z_correction_factor)
    return log_pred


