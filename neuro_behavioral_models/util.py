import jax.numpy as jnp
import jax.random as jr
import jax
import optax
import jaxopt
import tqdm
from tensorflow_probability.substrates.jax import distributions as tfd
from jaxtyping import Array, Float, Int, PyTree, Bool
from typing import Tuple, Union, Callable
from scipy.optimize import linear_sum_assignment
from functools import partial
from jax_moseq.utils import psd_inv

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


def center_embedding(k: int) -> Float[Array, "k k-1"]:
    """Generate an orthonormal matrix that embeds R^(k-1) into the space of 0-sum vectors in R^k."""
    # using numpy.linalg.svd because jax version crashes on windows
    return jnp.linalg.svd(jnp.eye(k) - jnp.ones((k, k)) / k)[0][:, :-1]


def lower_dim(arr, axis=0):
    """Lower dimension in specified axis by projecting onto the space of 0-sum vectors."""
    arr = jnp.moveaxis(arr, axis, 0)
    k, *shape = arr.shape
    arr = arr.reshape(k, -1)
    arr = center_embedding(k).T @ arr
    arr = arr.reshape(k - 1, *shape)
    arr = jnp.moveaxis(arr, 0, axis)
    return arr


def raise_dim(arr, axis=0):
    """Raise dimension in specified axis by embedding into the space of 0-sum vectors."""
    arr = jnp.moveaxis(arr, axis, 0)
    k, *shape = arr.shape
    arr = arr.reshape(k, -1)
    arr = center_embedding(k + 1) @ arr
    arr = arr.reshape(k + 1, *shape)
    arr = jnp.moveaxis(arr, 0, axis)
    return arr


def gradient_descent(loss_fn, init_params, learning_rate=1e-3, num_iters=100):
    """
    Run gradient descent to minimize a loss function.

    Args:
        loss_fn: Objective function.
        init_params: Initial value of parameters to be estimated.
        optimizer: Optimizer.
        num_iters: Number of iterations.

    Returns:
        params: Optimized parameters.
        losses: Losses recorded at each epoch.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(init_params)
    loss_grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = loss_grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    params = init_params
    with tqdm.trange(num_iters) as pbar:
        for i in pbar:
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
    trans_probs: Union[
        Float[Array, "n_states n_states"], Float[Array, "n_timesteps n_states n_states"]
    ],
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
    if trans_probs.ndim == 2:
        trans_probs = jnp.repeat(trans_probs[na, ...], n_timesteps, axis=0)
    log_trans_probs = jnp.log(trans_probs)
    init_state = jr.categorical(seeds[0], jnp.ones(n_states) / n_states)

    def step(state, args):
        seed, logT = args
        next_state = jr.categorical(seed, logT[state])
        return next_state, next_state

    _, states = jax.lax.scan(step, init_state, (seeds[1:], log_trans_probs))
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


def sample_hmc(
    seed: Float[Array, "2"],
    log_prob_fn: Callable,
    init_params: PyTree,
    num_leapfrog_steps: Int = 3,
    step_size: Float = 0.001,
    num_results: Int = 1,
    num_burnin_steps: Int = 100,
) -> Tuple[PyTree, PyTree]:
    """Sample using Hamiltonian Monte Carlo."""
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )
    params, _, kernel_state = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=init_params,
        kernel=hmc_kernel,
        seed=seed,
        trace_fn=None,
        return_final_kernel_results=True,
    )
    return params, kernel_state


@partial(jax.jit, static_argnums=(1,))
def sample_laplace(
    seed: Float[Array, "2"],
    log_prob_fn: Callable,
    init_params: PyTree,
) -> PyTree:
    """Sample using Laplace approximation."""

    # find the mode of the posterior
    solver = jaxopt.GradientDescent(fun=lambda x: -log_prob_fn(x))
    opt = solver.run(init_params)
    mode = opt.params

    # calculate covariance matrix from hessian at mode
    mode, unravel_fn = jax.flatten_util.ravel_pytree(mode)
    ll_fn = lambda x: log_prob_fn(unravel_fn(x))
    hessian_at_mode = jax.hessian(ll_fn)(mode)
    covariance_matrix = psd_inv(-hessian_at_mode, diagonal_boost=1e-2)

    # sample from laplace approximation
    x = jr.multivariate_normal(seed, mean=mode, cov=covariance_matrix)
    return unravel_fn(x)
