import jax.numpy as jnp
import jax.random as jr
import jax
import optax
import tqdm
from tensorflow_probability.substrates.jax import distributions as tfd

from jaxtyping import Array, Float, Int
from typing import Tuple
from scipy.optimize import linear_sum_assignment

na = jnp.newaxis


@jax.vmap
def logits_to_probs(
    logits: Float[Array, "n_categories-1"]
) -> Float[Array, "n_categories"]:
    """Convert logits to probabilities."""
    logits = jnp.concatenate([logits, jnp.zeros(1)])
    return jax.nn.softmax(logits)


@jax.vmap
def probs_to_logits(
    probs: Float[Array, "n_categories"],
    pseudo_count: Float = 1e-8,
) -> Float[Array, "n_categories-1"]:
    """Convert probabilities to logits."""
    log_probs = jnp.log(probs + pseudo_count)
    return log_probs[:-1] - log_probs[-1]


def normal_inverse_gamma_posterior(
    seed: Float[Array, "2"],
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


def gradient_descent(
    loss_fn, params, learning_rate=1e-3, num_iters=100, key=jr.PRNGKey(0)
):
    """
    Run gradient descent to minimize a loss function.

    Args:
        loss_fn: Objective function.
        params: Initial value of parameters to be estimated.
        optimizer: Optimizer.
        num_iters: Number of iterations.
        key: RNG key.

    Returns:
        params: Optimized parameters.
        losses: Losses recorded at each epoch.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    loss_grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = loss_grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    with tqdm.trange(num_iters) as pbar:
        for i in pbar:
            key, subkey = jr.split(key)
            params, opt_state, loss = train_step(params, opt_state)
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})

    return params, jnp.array(losses)


def sample_multinomial(
    seed: Float[Array, "2"],
    n: Int,
    p: Float[jnp.ndarray, "n_categories"],
) -> Int[Array, "n_categories"]:
    return tfd.Multinomial(n, probs=p).sample(seed=seed)


def sample_gamma(
    seed: Float[Array, "2"],
    a: Float,
    b: Float,
) -> Float:
    return jr.gamma(seed, a) / b


def sample_inv_gamma(
    seed: Float[Array, "2"],
    a: Float,
    b: Float,
) -> Float:
    return 1.0 / sample_gamma(seed, a, b)


def simulate_hmm_states(
    seed: Float[Array, "2"],
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
