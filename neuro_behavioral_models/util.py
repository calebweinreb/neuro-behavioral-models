import jax.numpy as jnp
import jax.random as jr
import jax
from jax.scipy.special import gammaln, logsumexp
from tensorflow_probability.substrates.jax import distributions as tfd
from dynamax.hidden_markov_model.inference import hmm_posterior_sample
from jaxtyping import Array, Float, Int
from typing import Tuple
from scipy.optimize import linear_sum_assignment


na = jnp.newaxis


def sample_multinomial(
    seed: jr.PRNGKey,
    n: Int,
    p: Float[jnp.ndarray, "n_categories"],
) -> Int[Array, "n_categories"]:
    return tfd.Multinomial(n, probs=p).sample(seed=seed)


def sample_gamma(
    seed: jr.PRNGKey,
    a: Float,
    b: Float,
) -> Float:
    return jr.gamma(seed, a) / b


def sample_inv_gamma(
    seed: jr.PRNGKey,
    a: Float,
    b: Float,
) -> Float:
    return 1.0 / sample_gamma(seed, a, b)


def normal_inverse_gamma_posterior(
    seed: jr.PRNGKey,
    mean: Float,
    sigmasq: Float,
    n: Int,
    lambda_: Float,
    alpha: Float,
    beta: Float,
) -> Tuple[Float, Float]:
    """
    Sample posterior mean and variance given normal-inverse gamma prior.

    Args:
        seed: random seed
        mean: sample mean
        sigmasq: sample variance
        n: number of data points
        lambda_: strength of prior
        alpha: inverse gamma shape parameter
        beta: inverse gamma rate parameter

    Returns:
        mu: posterior mean
        sigma: posterior variance
    """
    seeds = jr.split(seed, 2)
    mean = jnp.nan_to_num(mean)
    sigmasq = jnp.nan_to_num(sigmasq)
    lambda_n = lambda_ + n
    alpha_n = alpha + n / 2
    beta_n = beta + 0.5 * n * sigmasq + 0.5 * n * lambda_ * (mean**2) / lambda_n
    sigma = sample_inv_gamma(seeds[0], alpha_n, beta_n)
    mu = jr.normal(seeds[1]) * jnp.sqrt(sigmasq / lambda_n) + mean
    return mu, sigma


def multinomial_log_prob(
    probs: Float[Array, "n_categories"],
    counts: Int[Array, "... n_categories"],
) -> Float[Array, "..."]:
    """
    Compute the log probability of counts in a multinomial distribution.

    Args:
        probs: Probabilities for each category.
        counts: Observed counts for each category, with arbitrary batch dimensions.

    Returns:
        Log probabilities with the same batch dimensions as counts.
    """
    return (
        gammaln(counts.sum(-1) + 1)
        - gammaln(counts + 1).sum(-1)
        + (counts * jnp.log(probs)).sum(-1)
    )


def sample_hmm_states(
    seed: jr.PRNGKey,
    log_likelihoods: Float[Array, "n_timesteps n_states"],
    trans_probs: Float[Array, "n_states n_states"],
    mask: Int[Array, "n_timesteps"],
) -> Tuple[Float, Int[Array, "n_timesteps"]]:
    """Sample state sequences in a Markov chain.

    Args:
        seed: random seed
        log_likelihoods: log likelihoods of observations given states
        trans_probs: transition probabilities between states
        mask: indicates which observations are valid

    Returns:
        L: log probability of the sampled state sequence
        z: sampled state sequence
    """
    n_states = trans_probs.shape[0]
    initial_distribution = jnp.ones(n_states) / n_states
    log_likelihoods -= logsumexp(log_likelihoods, axis=-1, keepdims=True)

    masked_log_likelihoods = log_likelihoods * mask[:, na]
    L, z = hmm_posterior_sample(
        seed, initial_distribution, trans_probs, masked_log_likelihoods
    )
    return L, z


def simulate_hmm_states(
    seed: jr.PRNGKey,
    trans_probs: Float[Array, "n_states n_states"],
    n_timesteps: Int,
) -> Int[Array, "n_timesteps"]:
    """Simulate a state sequence from in Markov chain.

    Args:
        seed: random seed
        trans_probs: transition probabilities between states
        n_timesteps: number of timesteps to simulate

    Returns:
        states: simulated state sequence
    """
    seeds = jr.split(seed, n_timesteps + 1)
    n_states = trans_probs.shape[0]
    log_trans_probs = jnp.log(trans_probs)
    init_state = jr.categorical(seeds[0], jnp.ones(n_states) / n_states)

    def step(state, seed):
        next_state = jr.categorical(seed, log_trans_probs[state])
        return next_state, next_state

    _, states = jax.lax.scan(step, init_state, seeds[1:])
    return states


def compare_states(
    true_states: Int[Array, "n_sessions n_timesteps"],
    pred_states: Int[Array, "n_sessions n_timesteps"],
    n_states: Int,
) -> Tuple[Int[Array, "n_states n_states"], Int[Array, "n_states"], Float]:
    """Compare true and predicted states.

    Args:
        true_states: true state sequences
        pred_states: predicted state sequences
        n_states: number of states

    Returns:
        confusion_matrix: confusion matrix
        optimal_permutation: optimal permutation of predicted states
        accuracy: proportion of correct labels (after optimal permutation)
    """
    confusion = jnp.zeros((n_states, n_states)).at[pred_states, true_states].add(1)
    optimal_perm = linear_sum_assignment(-confusion.T)[1]
    accuracy = confusion[optimal_perm, jnp.arange(n_states)].sum() / true_states.size
    confusion = confusion / confusion.sum(axis=1, keepdims=True)
    return confusion, optimal_perm, accuracy
