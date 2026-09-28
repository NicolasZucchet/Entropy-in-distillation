"""The toy model of the paper (Section 3.1 and Appendix C.6).

A linear softmax student pi(. | i) = softmax(W e_i + b) over d = 64 tokens is distilled, with
full-batch Adam, into teachers with Gaussian logits on n fixed unit-norm Gaussian context
embeddings e_i of dimension h = 128. The seeds of one sweep point are trained together in one
vmapped scan; each seed redraws the contexts, the teachers and the initialization.
"""
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

VOCAB, EMBED = 64, 128


# --- Contexts and teachers ---------------------------------------------------------------------

def _softmax_rows(Z, taus):
    L = Z / taus[:, None]
    L -= L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def _entropy_rows(P):
    safe = np.where(P > 0, P, 1.0)
    return -np.sum(np.where(P > 0, P * np.log(safe), 0.0), axis=1)


def sample_teachers(rng, perplexity, n):
    """n teachers softmax(z / tau) with Gaussian logits z and tau bisected so that H = log(ppl).

    `perplexity` is one value shared by every teacher or an (n,) array of per-teacher values.
    """
    Z = rng.standard_normal((n, VOCAB)).astype(np.float64)
    H_target = np.log(np.asarray(perplexity, dtype=np.float64))
    if H_target.ndim == 0:  # closed forms at the two ends of the scale
        if H_target <= 0.0:
            one_hot = np.zeros((n, VOCAB), dtype=np.float32)
            one_hot[np.arange(n), np.argmax(Z, axis=1)] = 1.0
            return one_hot
        if H_target >= np.log(VOCAB) - 1e-8:
            return np.ones((n, VOCAB), dtype=np.float32) / VOCAB
    H_target = np.broadcast_to(H_target, (n,))
    lo, hi = np.full(n, 1e-6), np.full(n, 1e4)
    for _ in range(64):
        mid = (lo + hi) / 2
        H_mid = _entropy_rows(_softmax_rows(Z, mid))
        lo = np.where(H_mid < H_target, mid, lo)
        hi = np.where(H_mid >= H_target, mid, hi)
    return _softmax_rows(Z, (lo + hi) / 2).astype(np.float32)


class Student(nn.Module):
    """Linear layer with LeCun-normal weights and zero bias."""

    @nn.compact
    def __call__(self, x):
        return nn.Dense(VOCAB)(x)


STUDENT = Student()


@dataclass
class Task:
    x: jnp.ndarray        # (n, EMBED) context embeddings
    T: jnp.ndarray        # (n, VOCAB) teacher distributions
    weights: jnp.ndarray  # (n,) loss weights, all 1/n
    params0: dict         # student initialization


def make_task(n, seed, perplexity=None, ppl_max=16.0):
    """Contexts, teachers and initialization for one seed.

    With `perplexity=None` every context draws its own teacher perplexity log-uniformly on
    [1, ppl_max]; otherwise every teacher has that perplexity.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, EMBED)).astype(np.float32)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    if perplexity is None:
        rng.uniform(0.0, 0.0, size=n)  # unused draw, kept so the random stream matches data/
        perplexity = np.exp(rng.uniform(0.0, np.log(ppl_max), size=n))
    T = sample_teachers(rng, perplexity, n)
    weights = jnp.full((n,), 1.0 / n, dtype=jnp.float32)
    params0 = STUDENT.init(jax.random.PRNGKey(seed), jnp.zeros((1, EMBED)))
    return Task(jnp.array(x), jnp.array(T), weights, params0)


# --- Divergences, one value per context --------------------------------------------------------

def forward_kl(logits, T):
    """KL(T || pi)."""
    log_pi = jax.nn.log_softmax(logits, axis=-1)
    return jnp.sum(T * (jnp.log(T + 1e-10) - log_pi), axis=-1)


def reverse_kl(logits, T):
    """KL(pi || T)."""
    log_pi = jax.nn.log_softmax(logits, axis=-1)
    pi = jnp.exp(log_pi)
    return jnp.sum(pi * (log_pi - jnp.log(T + 1e-10)), axis=-1)


def jsd(alpha):
    """Generalized Jensen-Shannon divergence alpha KL(T || m) + (1 - alpha) KL(pi || m),
    m = alpha T + (1 - alpha) pi; alpha = 0 and 1 are the forward and reverse KL (their limits)."""
    if alpha == 0.0:
        return forward_kl
    if alpha == 1.0:
        return reverse_kl

    def divergence(logits, T):
        log_pi = jax.nn.log_softmax(logits, axis=-1)
        pi = jnp.exp(log_pi)
        log_T = jnp.log(T + 1e-10)
        log_m = jnp.log(alpha * T + (1.0 - alpha) * pi + 1e-10)
        fwd = jnp.sum(T * (log_T - log_m), axis=-1)
        rev = jnp.sum(pi * (log_pi - log_m), axis=-1)
        return alpha * fwd + (1.0 - alpha) * rev
    return divergence


def _add_tail_bucket(log_probs):
    # The clamp keeps log(1 - mass) finite when the window holds essentially all the mass.
    log_s = jnp.clip(jax.nn.logsumexp(log_probs, axis=-1, keepdims=True), a_max=-1e-7)
    return jnp.concatenate([log_probs, jnp.log(-jnp.expm1(log_s))], axis=-1)


def topk_jsd(alpha, k, tail, window):
    """Generalized JSD restricted to the top-k tokens of the student or teacher (`window`), as in
    the self-distillation reference implementation; the rest of the mass goes into one extra
    token (`tail=True`) or is dropped by renormalizing both sides (`tail=False`)."""
    def divergence(logits, T):
        log_pi = jax.nn.log_softmax(logits, axis=-1)
        log_T = jnp.log(T + 1e-10)
        if k < logits.shape[-1]:
            idx = jax.lax.top_k(logits if window == 'student' else T, k)[1]
            log_pi = jnp.take_along_axis(log_pi, idx, axis=-1)
            log_T = jnp.take_along_axis(log_T, idx, axis=-1)
            if tail:
                log_pi, log_T = _add_tail_bucket(log_pi), _add_tail_bucket(log_T)
            else:
                log_pi = log_pi - jax.nn.logsumexp(log_pi, axis=-1, keepdims=True)
                log_T = log_T - jax.nn.logsumexp(log_T, axis=-1, keepdims=True)
        if alpha == 0.0:
            return jnp.sum(jnp.exp(log_T) * (log_T - log_pi), axis=-1)
        if alpha == 1.0:
            return jnp.sum(jnp.exp(log_pi) * (log_pi - log_T), axis=-1)
        pi, teacher = jnp.exp(log_pi), jnp.exp(log_T)
        log_m = jnp.log(alpha * teacher + (1.0 - alpha) * pi + 1e-10)
        forward = jnp.sum(teacher * (log_T - log_m), axis=-1)
        reverse = jnp.sum(pi * (log_pi - log_m), axis=-1)
        return alpha * forward + (1.0 - alpha) * reverse
    return divergence


# --- Training ----------------------------------------------------------------------------------

def make_train(x, T, weights, divergence, n_steps, record):
    """train(params0, lr): full-batch Adam on sum_i w_i D(T_i, pi_i), returning the final
    parameters and the per-step loss, plus the per-step average entropy gap H(pi) - H(T) when
    `record` is set."""
    # Computed when the trainer is built, outside the jitted scan; moving it inside changes the
    # results in the last bit.
    H_T = -jnp.sum(T * jnp.log(T + 1e-10), axis=-1)

    def loss_fn(params):
        return jnp.sum(weights * divergence(STUDENT.apply(params, x), T))

    def train(params0, lr):
        tx = optax.adam(lr)

        def step(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            if not record:
                return (params, opt_state), loss
            log_pi = jax.nn.log_softmax(STUDENT.apply(params, x), axis=-1)
            H_pi = -jnp.sum(jnp.exp(log_pi) * log_pi, axis=-1)
            return (params, opt_state), (loss, jnp.sum(weights * (H_pi - H_T)))

        (params, _), out = jax.lax.scan(step, (params0, tx.init(params0)), None, length=n_steps)
        return params, out
    return train


def _stack(tasks):
    return (jnp.stack([t.x for t in tasks]), jnp.stack([t.T for t in tasks]),
            jnp.stack([t.weights for t in tasks]),
            jax.tree_util.tree_map(lambda *a: jnp.stack(a), *[t.params0 for t in tasks]))


def train_to_end(tasks, divergence, lr, n_steps):
    """Train every task (one per seed) and return the final student distributions (seeds, n, d),
    the loss at the last step and the norm of W, one per seed."""
    def one(x, T, w, p0):
        params, loss = make_train(x, T, w, divergence, n_steps, False)(p0, jnp.float32(lr))
        pi = jax.nn.softmax(STUDENT.apply(params, x), axis=-1)
        return pi, loss[-1], jnp.linalg.norm(params['params']['Dense_0']['kernel'])

    pi, loss, wnorm = jax.jit(jax.vmap(one))(*_stack(tasks))
    return np.array(pi), np.array(loss), np.array(wnorm)


def entropy_trajectories(tasks, divergence, lr, n_steps):
    """Average entropy gap H(pi) - H(T) after every step, (seeds, n_steps)."""
    def one(x, T, w, p0):
        return make_train(x, T, w, divergence, n_steps, True)(p0, jnp.float32(lr))[1][1]

    return np.asarray(jax.jit(jax.vmap(one))(*_stack(tasks)))


def entropy_trajectory(task, divergence, lr, n_steps):
    """Average entropy gap after every step for a single task, (n_steps,)."""
    train = jax.jit(make_train(task.x, task.T, task.weights, divergence, n_steps, True))
    return np.asarray(train(task.params0, jnp.float32(lr))[1][1])


def teacher_entropy(task):
    """Average teacher entropy of a task."""
    return float(jnp.sum(task.weights * -jnp.sum(task.T * jnp.log(task.T + 1e-10), axis=-1)))


def entropy_gap(pi, T):
    """Per-context entropy gap H(pi) - H(T)."""
    H_pi = -np.sum(pi * np.log(pi + 1e-12), axis=-1)
    H_T = -np.sum(T * np.log(T + 1e-12), axis=-1)
    return H_pi - H_T
