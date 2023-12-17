import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm, dirichlet
from jaxtyping import Array, Float, Int
from jax.nn import softmax
import tqdm
import time
from typing import Tuple
from functools import partial
from dynamax.hidden_markov_model import (
    hmm_filter,
    hmm_smoother,
    hmm_posterior_mode,
    hmm_posterior_sample,
    parallel_hmm_posterior_sample,
)

from .util import (
    simulate_hmm_states,
    lower_dim,
    raise_dim,
    sample_laplace,
)
from jax_moseq.utils.transitions import resample_hdp_transitions, init_hdp_transitions

na = jnp.newaxis

"""
data = {
    "syllables": (n_sessions, n_timesteps, n_syllables),
    "mask": (n_sessions, n_timesteps),
}

states: (n_sessions, n_timesteps)

params = {
    "emission_base": (n_syllables, n_syllables-1),
    "emission_biases": (n_states-1, n_syllables-1),
    "trans_pi": (n_states, n_states),
    "trans_betas": (n_states, n_states),
}

hypparams = {
    "n_states": (,),
    "emission_base_sigma": (,),
    "emission_biases_sigma": (,),
    "trans_gamma": (,),
    "trans_alpha": (,),
    "trans_kappa": (,),
    "n_syllables"
}
"""


def estimate_emission_params(
    sufficient_stats: Float[Array, "n_states n_syllables n_syllables"],
) -> Tuple[
    Float[Array, "n_syllables n_syllables-1"], Float[Array, "n_states n_syllables-1"]
]:
    """Estimate emission parameters from transition counts."""
    logits = jnp.log(sufficient_stats + 1e-2)
    emission_base_est = lower_dim(logits.mean(0), 1)
    emission_biases_est = (logits - logits.mean(0)[na]).mean(1)
    emission_biases_est = lower_dim(lower_dim(emission_biases_est, 1), 0)
    return emission_base_est, emission_biases_est


def get_syllable_trans_probs(
    emission_base: Float[Array, "n_syllables n_syllables-1"],
    emission_biases: Float[Array, "n_states n_syllables-1"],
) -> Float[Array, "n_states n_syllables n_syllables"]:
    """Compute transition probabilities between syllables."""
    emission_base = raise_dim(emission_base, 1)
    emission_biases = raise_dim(raise_dim(emission_biases, 0), 1)
    logits = emission_base[na] + emission_biases[:, na]
    return softmax(logits, axis=-1)


def obs_log_likelihoods(
    data: dict,
    params: dict,
) -> Float[Array, "n_sessions n_timesteps n_states"]:
    """Compute log likelihoods of observations for each hidden state."""

    n_sessions = data["syllables"].shape[0]
    n_states = params["trans_pi"].shape[0]

    log_syllable_trans_probs = jnp.log(
        get_syllable_trans_probs(
            params["emission_base"],
            params["emission_biases"],
        )
    )
    log_likelihoods = jax.vmap(
        lambda T: T[data["syllables"][:, :-1], data["syllables"][:, 1:]]
    )(log_syllable_trans_probs)

    log_likelihoods = jnp.concatenate(
        [jnp.zeros((n_states, n_sessions, 1)), log_likelihoods], axis=2
    )
    return log_likelihoods.transpose((1, 2, 0)) * data["mask"][:, :, na]


def log_params_prob(
    params: dict,
    hypparams: dict,
) -> Float:
    """Compute the log probability of the parameters based on their priors."""

    n_states = params["trans_pi"].shape[0]

    # prior on emission base parameters
    emission_base_log_prob = norm.logpdf(
        params["emission_base"] / hypparams["emission_base_sigma"]
    ).sum()

    # prior on emission bias parameters
    emission_biases_log_prob = norm.logpdf(
        params["emission_biases"] / hypparams["emission_biases_sigma"]
    ).sum()

    # prior on transition parameters
    beta_conc = jnp.full(n_states, hypparams["trans_gamma"] / n_states)
    beta_log_prob = dirichlet.logpdf(params["trans_betas"], beta_conc)

    pi_conc = (
        hypparams["trans_kappa"] * jnp.eye(n_states)
        + hypparams["trans_alpha"] * params["trans_betas"]
    )
    pi = (params["trans_pi"] + 1e-8) / (params["trans_pi"] + 1e-8).sum(1)[:, None]
    pi_log_prob = jax.vmap(dirichlet.logpdf)(pi, pi_conc).sum()

    return (
        emission_base_log_prob + emission_biases_log_prob + beta_log_prob + pi_log_prob
    )


def log_joint_prob(
    data: dict,
    params: dict,
    hypparams: dict,
) -> Float:
    """Compute the log joint probability of the data and parameters."""
    return marginal_loglik(data, params) + log_params_prob(params, hypparams)


@partial(jax.jit, static_argnums=(3,))
def resample_states(
    seed: Float[Array, "2"],
    data: dict,
    params: dict,
    parallel: bool = False,
) -> Tuple[Int[Array, "n_sessions n_timesteps"], Float]:
    """Resample hidden states from their posterior distribution.

    Args:
        seed: random seed
        data: data dictionary
        params: parameters dictionary
        parallel: whether to use parallel message passing

    Returns:
        states: resampled hidden states
        marginal_loglik: marginal log likelihood of the data
    """
    n_states = params["trans_pi"].shape[0]
    seeds = jr.split(seed, data["syllables"].shape[0])

    sample_fn = parallel_hmm_posterior_sample if parallel else hmm_posterior_sample
    marginal_logliks, states = jax.vmap(sample_fn, in_axes=(0, None, None, 0))(
        seeds,
        jnp.ones(n_states) / n_states,
        params["trans_pi"],
        obs_log_likelihoods(data, params),
    )
    return states, marginal_logliks.sum()


def fit_gibbs(
    data: dict,
    hypparams: dict,
    init_params: dict,
    init_states: Int[Array, "n_sessions n_timesteps"] = None,
    seed: Float[Array, "2"] = jr.PRNGKey(0),
    num_iters: Int = 100,
    parallel: bool = False,
) -> Tuple[dict, Float[Array, "num_iters"]]:
    """Fit a model using Gibbs sampling.

    Args:
        data: data dictionary
        hypparams: hyperparameters dictionary
        init_params: initial parameters directionary
        init_states: initial hidden states (optional)
        seed: random seed
        num_iters: number of iterations
        parallel: whether to use parallel message passing

    Returns:
        params: fitted parameters dictionary
        log_joints: log joint probability of the data and parameters recorded at each iteration
    """
    if init_states is None:
        states, _ = resample_states(seed, data, init_params, parallel)

    log_joints = []
    params = init_params
    for _ in tqdm.trange(num_iters):
        seed, subseed = jr.split(seed)
        params, gd_losses = resample_params(subseed, data, params, states, hypparams)
        states, marginal_loglik = resample_states(seed, data, params, parallel)
        log_joints.append(marginal_loglik + log_params_prob(params, hypparams))
    return params, states, jnp.array(log_joints)


def initialize_params(
    data: dict,
    hypparams: dict,
    states: Int[Array, "n_sessions n_timesteps"] = None,
    seed: Float[Array, "2"] = jr.PRNGKey(0),
) -> dict:
    """Initialize parameters by sampling from their prior distribution or using
    provided states.

    Args:
        data: data dictionary
        hypparams: hyperparameters dictionary
        states: states used for initializing the parameters (optional)
        seed: random seed
    """
    if states is not None:
        params = resample_params(seed, data, states, hypparams)
    else:
        params = random_params(seed, hypparams)
    return params


def fit_gradient_descent(
    data: dict,
    hypparams: dict,
    init_params: dict,
    num_iters: Int = 100,
    learning_rate: Float = 1e-3,
) -> Tuple[dict, Float[Array, "num_iters"]]:
    """Fit a model using gradient descent.

    Args:
        data: data dictionary
        hypparams: hyperparameters dictionary
        init_params: initial parameters directionary
        num_iters: number of iterations
        learning_rate: learning rate for gradient descent

    Returns:
        params: fitted parameters dictionary
        log_joints: log joint probability of the data and parameters recorded at each iteration
    """
    loss_fn = lambda params: -log_joint_prob(data, params, hypparams)
    params, losses = gradient_descent(loss_fn, init_params, learning_rate, num_iters)
    log_joints = -losses
    return params, log_joints


def marginal_loglik(
    data: dict,
    params: dict,
) -> Float[Array, "n_sessions n_timesteps n_states"]:
    """Estimate marginal log likelihood of the data"""
    n_states = params["trans_pi"].shape[0]
    mll = jax.vmap(hmm_filter, in_axes=(None, None, 0))(
        jnp.ones(n_states) / n_states,
        params["trans_pi"],
        obs_log_likelihoods(data, params),
    ).marginal_loglik.sum()
    return mll


def smoothed_states(
    data: dict,
    params: dict,
) -> Float[Array, "n_sessions n_timesteps n_states"]:
    """Estimate marginals of hidden states using forward-backward algorithm."""
    n_states = params["trans_pi"].shape[0]
    return jax.vmap(hmm_smoother, in_axes=(None, None, 0))(
        jnp.ones(n_states) / n_states,
        params["trans_pi"],
        obs_log_likelihoods(data, params),
    ).smoothed_probs


def predicted_states(
    data: dict,
    params: dict,
) -> Float[Array, "n_sessions n_timesteps n_states"]:
    """Predict hidden states using Viterbi algorithm."""
    n_states = params["trans_pi"].shape[0]
    return jax.vmap(hmm_posterior_mode, in_axes=(None, None, 0))(
        jnp.ones(n_states) / n_states,
        params["trans_pi"],
        obs_log_likelihoods(data, params),
    )


def random_params(
    seed: Float[Array, "2"],
    hypparams: dict,
) -> dict:
    """Generate random model parameters.

    emission_base ~ Normal(0, emission_base_sigma)
    emission_biases ~ Normal(0, emission_biases_sigma)
    trans_probs ~ Dirichlet(trans_beta + trans_kappa * I)

    Args:
        seed: random seed
        hypparams: hyperparameters dictionary

    Returns:
        params: parameters dictionary
    """
    n_syllables = hypparams["n_syllables"]
    n_states = hypparams["n_states"]
    seeds = jr.split(seed, 3)

    emission_base = (
        jr.normal(seeds[0], shape=(n_syllables, n_syllables - 1))
        * hypparams["emission_base_sigma"]
    )

    emission_biases = (
        jr.normal(seeds[1], shape=(n_states - 1, n_syllables - 1))
        * hypparams["emission_biases_sigma"]
    )

    trans_betas, trans_probs = init_hdp_transitions(
        seeds[3],
        n_states,
        hypparams["trans_alpha"],
        hypparams["trans_kappa"],
        hypparams["trans_gamma"],
    )

    return {
        "emission_base": emission_base,
        "emission_biases": emission_biases,
        "trans_pi": trans_probs,
        "trans_betas": trans_betas,
    }


def simulate(seed, params, n_timesteps, n_sessions):
    """Simulate data from the model."""
    seeds = jr.split(seed, 3)

    states = jax.vmap(simulate_hmm_states, in_axes=(0, None, None))(
        jr.split(seeds[0], n_sessions),
        params["trans_pi"],
        n_timesteps,
    )
    syllable_trans_probs = get_syllable_trans_probs(
        params["emission_base"],
        params["emission_biases"],
    )[states]

    syllables = jax.vmap(simulate_hmm_states, in_axes=(0, 0, None))(
        jr.split(seeds[1], n_sessions), syllable_trans_probs, n_timesteps
    )
    return states, syllables


def resample_params(
    seed: Float[Array, "2"],
    data: dict,
    params: dict,
    states: Int[Array, "n_sessions n_timesteps"],
    hypparams: dict,
) -> Tuple[dict, Float[Array, "gradient_descent_iters"]]:
    """Resample parameters from their posterior distribution. Emission parameters are
    resampled using a Laplace approximation; the mode is found using gradient descent.

    Args:
        seed: random seed
        data: data dictionary
        params: parameters dictionary
        states: hidden states
        hypparams: hyperparameters dictionary

    Returns:
        params: parameters dictionary
        losses: losses recorded during gradient descent
    """
    seeds = jr.split(seed, 2)

    emission_params, gd_losses = resample_emission_params(
        seeds[1],
        data["syllables"],
        data["mask"],
        states,
        hypparams["n_states"],
        hypparams["n_syllables"],
        hypparams["emission_base_sigma"],
        hypparams["emission_biases_sigma"],
        hypparams["emission_gd_iters"],
        hypparams["emission_gd_lr"],
    )
    trans_betas, trans_probs = resample_hdp_transitions(
        seeds[2],
        states,
        data["mask"],
        params["trans_betas"],
        hypparams["trans_alpha"],
        hypparams["trans_kappa"],
        hypparams["trans_gamma"],
    )
    params = {
        "emission_base": emission_params[0],
        "emission_biases": emission_params[1],
        "trans_pi": trans_probs,
        "trans_betas": trans_betas,
    }
    return params, gd_losses


@partial(jax.jit, static_argnums=(4, 5, 8))
def resample_emission_params(
    seed: Float[Array, "2"],
    syllables: Int[Array, "n_sessions n_timesteps"],
    mask: Int[Array, "n_sessions n_timesteps"],
    states: Int[Array, "n_sessions n_timesteps"],
    n_states: int,
    n_syllables: int,
    emission_base_sigma: Float,
    emission_biases_sigma: Float,
    gradient_descent_iters: Int = 100,
    gradient_descent_lr: Float = 1e-3,
) -> Tuple[
    Tuple[
        Float[Array, "n_syllables n_syllables-1"],
        Float[Array, "n_states n_syllables-1"],
    ],
    Float[Array, "gradient_descent_iters"],
]:
    """Resample emission parameters from their posterior distribution.

    Args:
        seed: random seed
        syllables: syllable observations
        mask: mask of valid observations
        states: hidden states
        n_states: number of hidden states
        n_syllables: number of syllables
        emission_base_sigma: emission base standard deviation
        emission_biases_sigma: emission biases standard deviation

    Returns:
        emission_base: posterior emission base parameters
        emission_biases: posterior emission biases parameters
        losses: losses recorded during gradient descent
    """
    sufficient_stats = (
        jnp.zeros((n_states, n_syllables, n_syllables))
        .at[states[:, 1:], syllables[:, :-1], syllables[:, 1:]]
        .add(mask[:, 1:])
    )

    def log_prob_fn(args):
        emission_base, emission_biases = args
        syllable_trans_probs = get_syllable_trans_probs(emission_base, emission_biases)

        emission_base = raise_dim(emission_base, 1)
        emission_biases = raise_dim(raise_dim(emission_biases, 0), 1)
        prior_log_prob = (
            norm.logpdf(emission_base / emission_base_sigma).sum()
            + norm.logpdf(emission_biases / emission_biases_sigma).sum()
        )
        syllables_log_prob = (jnp.log(syllable_trans_probs) * sufficient_stats).sum()
        return prior_log_prob + syllables_log_prob

    init_emission_params = estimate_emission_params(sufficient_stats)
    (emission_base, emission_biases), losses = sample_laplace(
        seed,
        log_prob_fn,
        init_emission_params,
        gradient_descent_iters,
        gradient_descent_lr,
    )
    return (emission_base, emission_biases), losses
