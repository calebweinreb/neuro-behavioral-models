import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm, gamma, dirichlet, multinomial
from jaxtyping import Array, Float, Int, Bool
from typing import Tuple
from functools import partial
from jax.scipy.special import logsumexp
from dynamax.hidden_markov_model.inference import (
    hmm_filter,
    hmm_smoother,
    hmm_posterior_mode,
)

from .util import (
    sample_inv_gamma,
    simulate_hmm_states,
    sample_multinomial,
    normal_inverse_gamma_posterior,
    logits_to_probs,
    probs_to_logits,
)

na = jnp.newaxis

"""
data = {
    "neural_obs": (n_sessions, n_timesteps, n_features),
    "behavior_obs": (n_sessions, n_timesteps, n_syllables),
    "mask": (n_sessions, n_timesteps),
}

states: (n_sessions, n_timesteps)

params = {
    "neural_means": (n_sessions, n_states, n_features),
    "neural_vars": (n_sessions, n_states, n_features),
    "behavior_probs": (n_states, n_syllables-1),
    "trans_probs": (n_states, n_states-1),
}

hypparams = {
    "n_states": (,),
    "neural_lambda": (,),
    "neural_alpha": (,),
    "neural_beta": (,),
    "behavior_beta": (,),
    "trans_beta": (,),
    "trans_kappa": (,),
    "n_features",
    "n_syllables"
}

"""


def obs_log_likelihoods(
    data: dict,
    params: dict,
    ignore_neural_obs: Bool = False,
) -> Float[Array, "n_sessions n_timesteps n_states"]:
    """Compute log likelihoods of observations for each hidden state."""

    n_states = params["neural_means"].shape[1]
    n_syllables = data["behavior_obs"].shape[2]
    n_sessions, n_timesteps = data["neural_obs"].shape[:2]

    # log likelihoods of behavioral observations
    behavior_log_likelihoods = (
        jax.lax.map(
            partial(
                jax.vmap(multinomial.logpmf, in_axes=(0, 0, None)),
                data["behavior_obs"].reshape(-1, n_syllables),
                data["behavior_obs"].sum(-1).reshape(-1),
            ),
            logits_to_probs(params["behavior_probs"]),
        ).reshape(n_states, n_sessions, n_timesteps)
    ).transpose((1, 2, 0))

    # log likelihoods of neural observations
    neural_log_likelihoods = (
        norm.logpdf(
            data["neural_obs"][:, na, :, :],
            params["neural_means"][:, :, na, :],
            jnp.sqrt(params["neural_vars"])[:, :, na, :],
        )
        .sum(-1)
        .transpose((0, 2, 1))
    ) * (1 - ignore_neural_obs)

    log_likelihoods = behavior_log_likelihoods + neural_log_likelihoods
    return log_likelihoods * data["mask"][:, :, na]


def log_model_prob(
    data: dict,
    params: dict,
    hypparams: dict,
    ignore_neural_obs: Bool = False,
) -> Float:
    """Compute the log joint probability of the model."""

    n_states = params["neural_means"].shape[1]
    n_syllables = data["behavior_obs"].shape[2]

    # prior on neural emission parameters
    neural_means_log_prob = norm.logpdf(
        params["neural_means"],
        jnp.zeros_like(params["neural_means"]),
        jnp.sqrt(params["neural_vars"] / hypparams["neural_lambda"]),
    ).sum()

    neural_vars_log_prob = gamma.logpdf(
        1 / params["neural_vars"],
        hypparams["neural_alpha"],
        scale=1 / hypparams["neural_beta"],
    ).sum()

    # prior on behavior emission parameters
    behavior_probs_log_prob = jax.vmap(dirichlet.logpdf, in_axes=(0, None))(
        logits_to_probs(params["behavior_probs"]),
        hypparams["behavior_beta"] * jnp.ones(n_syllables),
    ).sum()

    # prior on transition parameters
    trans_probs_log_prob = jax.vmap(dirichlet.logpdf)(
        logits_to_probs(params["trans_probs"]),
        jnp.eye(n_states) * hypparams["trans_kappa"] + hypparams["trans_beta"],
    ).sum()

    # marginalize over states
    obs_log_prob = jax.vmap(hmm_filter, in_axes=(None, None, 0))(
        jnp.ones(n_states) / n_states,
        logits_to_probs(params["trans_probs"]),
        obs_log_likelihoods(data, params, ignore_neural_obs),
    ).marginal_loglik.sum()

    return (
        neural_means_log_prob * (1 - ignore_neural_obs)
        + neural_vars_log_prob * (1 - ignore_neural_obs)
        + behavior_probs_log_prob
        + trans_probs_log_prob
        + obs_log_prob
    )


def fit_model(
    data: dict,
    hypparams: dict,
    params: dict = None,
    states: Int[Array, "n_sessions n_timesteps"] = None,
    seed: Float[Array, "2"] = jr.PRNGKey(0),
    num_iters: Int = 100,
    learning_rate: Float = 1e-3,
    ignore_neural_obs: Bool = False,
) -> Tuple[dict, Float[Array, "num_iters"]]:
    """Fit a model using gradient descent.

    Args:
        data: data dictionary
        hypparams: hyperparameters dictionary
        params: initial parameters directionary (optional)
        states: states used for initializing the parameters (optional)
        seed: random seed used for parameter initialization and gradient descent
        num_iters: number of iterations
        learning_rate: learning rate for gradient descent
        ignore_neural_obs: whether to ignore the neural observations

    Returns:
        params: fitted parameters dictionary
        losses: losses recorded at each iteration
    """
    if states is not None:
        params = resample_params(seed, data, states, hypparams)
    elif params is None:
        n_sessions = data["behavior_obs"].shape[0]
        params = random_params(seed, hypparams, n_sessions)

    loss_fn = lambda params: -log_model_prob(data, params, hypparams, ignore_neural_obs)
    params, losses = gradient_descent(loss_fn, params, learning_rate, num_iters, seed)
    return params, losses


def smoothed_states(
    data: dict,
    params: dict,
    ignore_neural_obs: Bool = False,
) -> Float[Array, "n_sessions n_timesteps n_states"]:
    """Estimate marginals of hidden states using forward-backward algorithm."""

    n_states = params["neural_means"].shape[1]
    return jax.vmap(hmm_smoother, in_axes=(None, None, 0))(
        jnp.ones(n_states) / n_states,
        logits_to_probs(params["trans_probs"]),
        obs_log_likelihoods(data, params, ignore_neural_obs),
    ).smoothed_probs


def predicted_states(
    data: dict,
    params: dict,
    ignore_neural_obs: Bool = False,
) -> Float[Array, "n_sessions n_timesteps n_states"]:
    """Predict hidden states using Viterbi algorithm."""

    n_states = params["neural_means"].shape[1]
    return jax.vmap(hmm_posterior_mode, in_axes=(None, None, 0))(
        jnp.ones(n_states) / n_states,
        logits_to_probs(params["trans_probs"]),
        obs_log_likelihoods(data, params, ignore_neural_obs),
    )


def random_params(
    seed: Float[Array, "2"],
    hypparams: dict,
    n_sessions: Int,
) -> dict:
    """Generate random model parameters.

    neural_vars ~ inverseGamma(neural_degs, neural_scale * I)
    neural_means ~ Normal(0, neural_vars / neural_lambda)
    behavior_probs ~ Dirichlet(behavior_beta)
    trans_probs ~ Dirichlet(trans_beta + trans_kappa * I)

    Args:
        seed: random seed
        hypparams: hyperparameters dictionary
        n_sessions: number of sessions

    Returns:
        params: parameters dictionary
    """
    n_features = hypparams["n_features"]
    n_syllables = hypparams["n_syllables"]
    n_states = hypparams["n_states"]
    seeds = jr.split(seed, 4)

    neural_vars = jax.vmap(sample_inv_gamma, in_axes=(0, None, None))(
        jr.split(seeds[0], n_sessions * n_states * n_features),
        hypparams["neural_alpha"],
        hypparams["neural_beta"],
    ).reshape(n_sessions, n_states, n_features)

    neural_means = jr.normal(
        seeds[1], shape=(n_sessions, n_states, n_features)
    ) * jnp.sqrt(neural_vars / hypparams["neural_lambda"])

    behavior_probs = jax.vmap(jr.dirichlet)(
        jr.split(seeds[2], n_states),
        jnp.ones((n_states, n_syllables)) * hypparams["behavior_beta"],
    )
    trans_probs = jax.vmap(jr.dirichlet)(
        jr.split(seeds[3], n_states),
        jnp.eye(n_states) * hypparams["trans_kappa"] + hypparams["trans_beta"],
    )
    return {
        "neural_means": neural_means,
        "neural_vars": neural_vars,
        "behavior_probs": probs_to_logits(behavior_probs),
        "trans_probs": probs_to_logits(trans_probs),
    }


def simulate(seed, params, n_timesteps, n_syllables_per_timestep):
    """Simulate data from the model."""
    seeds = jr.split(seed, 3)
    n_sessions, _, n_features = params["neural_means"].shape
    n_syllables = params["behavior_probs"].shape[1] + 1

    states = jax.vmap(simulate_hmm_states, in_axes=(0, None, None))(
        jr.split(seeds[0], n_sessions), params["trans_probs"], n_timesteps
    )
    neural_means = jnp.take_along_axis(params["neural_means"], states[:, :, na], 1)
    neural_vars = jnp.take_along_axis(params["neural_vars"], states[:, :, na], 1)
    neural_obs = (
        jr.normal(seeds[1], shape=(n_sessions, n_timesteps, n_features))
        * jnp.sqrt(neural_vars)
        + neural_means
    )
    behavior_obs = jax.vmap(sample_multinomial, in_axes=(0, None, 0))(
        jr.split(seeds[2], n_sessions * n_timesteps),
        n_syllables_per_timestep,
        logits_to_probs(params["behavior_probs"])[states.ravel()],
    ).reshape(n_sessions, n_timesteps, n_syllables)

    return states, behavior_obs, neural_obs


@partial(jax.jit, static_argnums=(4,))
def resample_neural_params(
    seed: Float[Array, "2"],
    neural_obs: Float[Array, "n_sessions n_timesteps n_features"],
    mask: Int[Array, "n_sessions n_timesteps"],
    states: Int[Array, "n_sessions n_timesteps"],
    n_states: int,
    lambda_: Float,
    alpha: Float,
    beta: Float,
) -> Tuple[
    Float[Array, "n_sessions n_states n_features"],
    Float[Array, "n_sessions n_states n_features"],
]:
    """Resample neural parameters from their posterior distribution.

    Args:
        seed: random seed
        neural_obs: neural observations
        mask: mask of valid observations
        states: hidden states
        n_states: number of hidden states
        lambda_: strength of prior
        alpha: inverse gamma shape parameter
        beta: inverse gamma rate parameter

    Returns:
        neural_means: posterior means of neural parameters
        neural_vars: posterior covariances of neural parameters
    """
    n_sessions, _, n_features = neural_obs.shape

    masks = jnp.eye(n_states)[states].transpose((0, 2, 1))[:, :, :, na]
    masks *= mask[:, na, :, na]

    neural_sample_means = (neural_obs[:, na] * masks).sum(2) / masks.sum(2)
    centered_obs = neural_obs[:, na] - neural_sample_means[:, :, na]
    neural_sample_vars = (centered_obs**2 * masks).sum(2) / masks.sum(2)
    neural_obs_counts = jnp.repeat(masks.sum(2)[:, :, na], n_features, axis=2)

    neural_means, neural_vars = jax.vmap(
        normal_inverse_gamma_posterior, in_axes=(0, 0, 0, 0, None, None, None)
    )(
        jr.split(seed, n_sessions * n_states * n_features),
        neural_sample_means.ravel(),
        neural_sample_vars.ravel(),
        neural_obs_counts.ravel(),
        lambda_,
        alpha,
        beta,
    )
    neural_means = neural_means.reshape(n_sessions, n_states, n_features)
    neural_vars = neural_vars.reshape(n_sessions, n_states, n_features)
    return neural_means, neural_vars


@partial(jax.jit, static_argnums=(4,))
def resample_behavior_params(
    seed: Float[Array, "2"],
    behavior_obs: Int[Array, "n_sessions n_timesteps n_syllables"],
    mask: Int[Array, "n_sessions n_timesteps"],
    states: Int[Array, "n_sessions n_timesteps"],
    n_states: int,
    beta: Float,
) -> Float[Array, "n_states n_syllables"]:
    """Resample behavior parameters from their posterior distribution.

    Args:
        seed: random seed
        behavior_obs: behavior observations as counts of syllables
        mask: mask of valid observations
        states: hidden states
        n_states: number of hidden states
        beta: Dirichlet concentration parameter

    Returns:
        behavior_probs: posterior behavior probabilities
    """
    masks = jnp.eye(n_states)[states].transpose((2, 0, 1))[:, :, :, na]
    masks *= mask[na, :, :, na]

    behavior_sample_counts = (behavior_obs[na] * masks).sum((1, 2))
    behavior_probs = jax.vmap(jr.dirichlet)(
        jr.split(seed, n_states),
        behavior_sample_counts + beta,
    )
    return probs_to_logits(behavior_probs)


@partial(jax.jit, static_argnums=(3,))
def resample_trans_probs(
    seed: Float[Array, "2"],
    mask: Int[Array, "n_sessions n_timesteps"],
    states: Int[Array, "n_sessions n_timesteps"],
    n_states: int,
    beta: Float,
    kappa: Float,
) -> Float[Array, "n_states n_states"]:
    """Resample transition probabilities from their posterior distribution.

    Args:
        seed: random seed
        mask: mask of valid observations
        states: hidden states
        n_states: number of hidden states
        beta: Dirichlet concentration parameter
        kappa: Dirichlet concentration parameter

    Returns:
        trans_probs: posterior transition probabilities
    """
    trans_counts = (
        jnp.zeros((n_states, n_states))
        .at[states[:, :-1], states[:, 1:]]
        .add(mask[:, :-1])
    )
    trans_probs = jax.vmap(jr.dirichlet)(
        jr.split(seed, n_states), trans_counts + beta + jnp.eye(n_states) * kappa
    )
    return probs_to_logits(trans_probs)


def resample_params(
    seed: Float[Array, "2"],
    data: dict,
    states: Int[Array, "n_sessions n_timesteps"],
    hypparams: dict,
) -> dict:
    """Resample parameters from their posterior distribution.

    Args:
        seed: random seed
        data: data dictionary
        states: hidden states
        hypparams: hyperparameters dictionary

    Returns:
        params: parameters dictionary
    """
    seeds = jr.split(seed, 3)
    neural_means, neural_vars = resample_neural_params(
        seeds[0],
        data["neural_obs"],
        data["mask"],
        states,
        hypparams["n_states"],
        hypparams["neural_lambda"],
        hypparams["neural_alpha"],
        hypparams["neural_beta"],
    )
    behavior_probs = resample_behavior_params(
        seeds[1],
        data["behavior_obs"],
        data["mask"],
        states,
        hypparams["n_states"],
        hypparams["behavior_beta"],
    )
    trans_probs = resample_trans_probs(
        seeds[2],
        data["mask"],
        states,
        hypparams["n_states"],
        hypparams["trans_beta"],
        hypparams["trans_kappa"],
    )
    params = {
        "neural_means": neural_means,
        "neural_vars": neural_vars,
        "trans_probs": trans_probs,
        "behavior_probs": behavior_probs,
    }
    return params
