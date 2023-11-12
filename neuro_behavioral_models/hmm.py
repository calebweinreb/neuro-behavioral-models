import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats.norm import logpdf as normal_logpdf
from jaxtyping import Array, Float, Int, Bool
from typing import Tuple

from neuro_behavioral_models.util import (
    sample_inv_gamma,
    normal_inverse_gamma_posterior,
    multinomial_log_prob,
    sample_hmm_states,
    simulate_hmm_states,
    sample_multinomial,
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
    "behavior_probs": (n_states, n_syllables),
    "trans_probs": (n_states, n_states),
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


def init_params(
    seed: jr.PRNGKey,
    hypparams: dict,
    n_sessions: Int,
) -> dict:
    """Initialize parameters for the model.

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
        "behavior_probs": behavior_probs,
        "trans_probs": trans_probs,
    }


def resample_neural_params(
    seed: jr.PRNGKey,
    neural_obs: Float[Array, "n_sessions n_timesteps n_features"],
    mask: Int[Array, "n_sessions n_timesteps"],
    states: Float[Array, "n_sessions n_timesteps"],
    n_states: Int,
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

    masks = (
        jnp.stack([states == i for i in range(n_states)], axis=1)[:, :, :, na]
        * mask[:, na, :, na]
    )
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


def resample_behavior_params(
    seed: jr.PRNGKey,
    behavior_obs: Int[Array, "n_sessions n_timesteps n_syllables"],
    mask: Int[Array, "n_sessions n_timesteps"],
    states: Float[Array, "n_sessions n_timesteps"],
    n_states: Int,
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
    masks = (
        jnp.stack([states == i for i in range(n_states)])[:, :, :, na]
        * mask[na, :, :, na]
    )
    behavior_sample_counts = (behavior_obs[na] * masks).sum((1, 2))
    behavior_probs = jax.vmap(jr.dirichlet)(
        jr.split(seed, n_states),
        behavior_sample_counts + beta,
    )
    return behavior_probs


def resample_trans_probs(
    seed: jr.PRNGKey,
    mask: Int[Array, "n_sessions n_timesteps"],
    states: Float[Array, "n_sessions n_timesteps"],
    n_states: Int,
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
    return trans_probs


def resample_params(
    seed: jr.PRNGKey,
    data: dict,
    states: Float[Array, "n_sessions n_timesteps"],
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


def resample_states(
    seed: jr.PRNGKey,
    data: dict,
    params: dict,
    ignore_neural_obs: Bool = False,
) -> Float[Array, "n_sessions n_timesteps"]:
    """Resample hidden states from their posterior distribution.

    Args:
        seed: random seed
        data: data dictionary
        params: parameters dictionary

    Returns:
        states: hidden states
    """
    n_sessions = data["neural_obs"].shape[0]
    log_likelihoods = jax.vmap(multinomial_log_prob, in_axes=(0, None))(
        params["behavior_probs"], data["behavior_obs"]
    ).transpose((1, 2, 0))

    if not ignore_neural_obs:
        log_likelihoods += (
            normal_logpdf(
                data["neural_obs"][:, na, :, :],
                params["neural_means"][:, :, na, :],
                jnp.sqrt(params["neural_vars"])[:, :, na, :],
            )
            .sum(-1)
            .transpose((0, 2, 1))
        )
    _, states = jax.vmap(sample_hmm_states, in_axes=(0, 0, None, 0))(
        jr.split(seed, n_sessions),
        log_likelihoods,
        params["trans_probs"],
        data["mask"],
    )
    return states


def init_model(seed, data, hypparams):
    """Initialize model parameters and latent states."""
    seeds = jr.split(seed, 3)
    n_sessions = data["neural_obs"].shape[0]
    params = init_params(seeds[0], hypparams, n_sessions)
    states = resample_states(seeds[1], data, params)
    return seeds[2], states, params


def resample_model(seed, data, states, params, hypparams):
    """Resample model parameters and latent states."""
    seeds = jr.split(seed, 3)
    params = resample_params(seeds[0], data, states, hypparams)
    states = resample_states(seeds[1], data, params)
    return seeds[2], states, params


def simulate(seed, params, n_timesteps, n_syllables_per_timestep):
    """Simulate data from the model."""
    seeds = jr.split(seed, 3)
    n_sessions, _, n_features = params["neural_means"].shape
    n_syllables = params["behavior_probs"].shape[1]

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
        params["behavior_probs"][states.ravel()],
    ).reshape(n_sessions, n_timesteps, n_syllables)

    return states, behavior_obs, neural_obs
