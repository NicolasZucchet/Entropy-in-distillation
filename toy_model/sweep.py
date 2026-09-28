"""The five sweeps behind the toy-model figures.

A sweep is a list of units, one point of the sweep with its five seeds trained together. Usage:

  python sweep.py run <sweep> [--task I --num-tasks N]   train units I, I+N, ... into outputs/<sweep>/
  python sweep.py merge <sweep>                          stack them into outputs/<sweep>.npz
  python sweep.py check <sweep> <unit>                   retrain one unit, compare with data/<sweep>.npz

Units already in outputs/<sweep>/ are skipped, so an interrupted run resumes. SMOKE=1 shrinks the
grids, the number of steps and the number of seeds for a quick end-to-end test.
"""
import argparse
import os

import numpy as np

import toy

SMOKE = os.environ.get('SMOKE') == '1'
LR = 0.1
N_STEPS = 300 if SMOKE else 40000
SEEDS = range(2 if SMOKE else 5)
# Trajectories are stored at 2000 log-spaced steps, enough for the log-scale step axes.
TRACE_STEPS = np.unique(np.geomspace(1, N_STEPS, 2000).astype(np.int64))

LOADS = sorted({int(round(v)) for v in np.geomspace(192, 67584, 21)})       # Figure 2b
PERPLEXITIES = np.geomspace(1.0, 32.0, 21)                                  # Figure 2c
ALPHAS = np.geomspace(0.001, 0.99, 21)                                      # Figure 2d
ALPHAS_SHOWN = [0.0, 0.01, 0.1, 0.5, 1.0]                                   # Figures 2e, 7, 8
HEAT_PERPLEXITIES = np.geomspace(1.5, 32.0, 10)                             # Figure 7
HEAT_LOADS = np.unique(np.geomspace(128, 4096, 8).astype(int))
TOPK_KS = [1, 4, 16, 64]                                                    # Figures 3, 8
N_DEFAULT, N_TOPK = 2048, 512
if SMOKE:
    LOADS, PERPLEXITIES, ALPHAS = [192, 512, 1024], PERPLEXITIES[::7], ALPHAS[::7]
    HEAT_PERPLEXITIES, HEAT_LOADS = HEAT_PERPLEXITIES[::4], HEAT_LOADS[:2]
    TOPK_KS, N_DEFAULT, N_TOPK = [1, 4, 64], 256, 128

SWEEPS = {
    'context_load': [dict(divergence=d, n=int(n)) for d in ('reverse', 'forward') for n in LOADS],
    'perplexity_gap': [dict(divergence=d, perplexity=float(p), n=N_DEFAULT)
                       for d in ('reverse', 'forward') for p in PERPLEXITIES],
    'jsd_family': [dict(kind='endpoint', alpha=float(a), n=N_DEFAULT) for a in ALPHAS]
                  + [dict(kind='trajectory', alpha=a, n=N_DEFAULT) for a in ALPHAS_SHOWN],
    'jsd_heatmap': [dict(alpha=a, perplexity=float(p), n=int(n))
                    for a in ALPHAS_SHOWN for p in HEAT_PERPLEXITIES for n in HEAT_LOADS],
    'topk': [dict(alpha=a, window=w, tail=t, k=k, n=N_TOPK)
             for a in ALPHAS_SHOWN for w in ('student', 'teacher') for t in (False, True)
             for k in TOPK_KS],
}


def endpoint(n, divergence, perplexity=None):
    """Seed-wise context-mean entropy gap and support size (tokens above 1e-4) at the end of
    training, with the final loss and |W|."""
    tasks = [toy.make_task(n, s, perplexity) for s in SEEDS]
    pi, loss, wnorm = toy.train_to_end(tasks, divergence, LR, N_STEPS)
    T = np.array(np.stack([t.T for t in tasks]))
    return dict(dH=toy.entropy_gap(pi, T).mean(axis=-1),
                support=np.sum(pi > 1e-4, axis=-1).mean(axis=-1), loss=loss, wnorm=wnorm)


def run_unit(sweep, u):
    """Train one unit and return its results as a dict of arrays."""
    kl = toy.reverse_kl if u.get('divergence') == 'reverse' else toy.forward_kl
    if sweep == 'context_load':
        return endpoint(u['n'], kl)
    if sweep == 'perplexity_gap':
        return endpoint(u['n'], kl, u['perplexity'])
    if sweep == 'jsd_heatmap':
        return endpoint(u['n'], toy.jsd(u['alpha']), u['perplexity'])
    if sweep == 'jsd_family' and u['kind'] == 'endpoint':
        return endpoint(u['n'], toy.jsd(u['alpha']))
    if sweep == 'jsd_family':  # one seed only
        task = toy.make_task(u['n'], SEEDS[0])
        H_T = toy.teacher_entropy(task)
        dH = toy.entropy_trajectory(task, toy.jsd(u['alpha']), LR, N_STEPS)
        return dict(H_T=H_T, steps=TRACE_STEPS, entropy=(dH + H_T)[TRACE_STEPS - 1])
    # topk: the loss only sees the k-token window, the entropy is over the full vocabulary
    tasks = [toy.make_task(u['n'], s) for s in SEEDS]
    H_T = np.array([toy.teacher_entropy(t) for t in tasks])
    divergence = toy.topk_jsd(u['alpha'], u['k'], u['tail'], u['window'])
    dH = toy.entropy_trajectories(tasks, divergence, LR, N_STEPS)
    return dict(H_T=H_T, steps=TRACE_STEPS, entropy=(dH + H_T[:, None])[:, TRACE_STEPS - 1])


def unit_path(sweep, i):
    return os.path.join('outputs', sweep, f'{i:04d}.npz')


def run(sweep, task, num_tasks):
    for i in range(task, len(SWEEPS[sweep]), num_tasks):
        path = unit_path(sweep, i)
        if os.path.exists(path):
            continue
        u = SWEEPS[sweep][i]
        print(f'{sweep} {i}/{len(SWEEPS[sweep])} {u}', flush=True)
        result = {**u, **run_unit(sweep, u)}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path + '.tmp.npz', **result)
        os.replace(path + '.tmp.npz', path)


def merge(sweep):
    """One file per sweep: key '<unit>/<field>' for every field of every unit."""
    arrays = {'n_steps': N_STEPS, 'n_seeds': len(SEEDS)}
    for i in range(len(SWEEPS[sweep])):
        with np.load(unit_path(sweep, i)) as f:
            arrays.update({f'{i}/{k}': f[k] for k in f.files})
    np.savez_compressed(os.path.join('outputs', f'{sweep}.npz'), **arrays)
    print(f'wrote outputs/{sweep}.npz')


def check(sweep, i):
    """Retrain unit i and print its largest difference to the committed data."""
    if SMOKE:
        raise SystemExit('check compares with the full-size sweeps in data/; unset SMOKE')
    from plot import load
    ref = load(sweep)[i]
    new = run_unit(sweep, SWEEPS[sweep][i])
    print(f'{sweep} unit {i}: {SWEEPS[sweep][i]}')
    for key, value in new.items():
        diff = np.max(np.abs(np.asarray(value, dtype=float) - np.asarray(ref[key], dtype=float)))
        print(f'  {key:8s} max |difference| = {diff:.3g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'merge', 'check'])
    parser.add_argument('sweep', choices=list(SWEEPS))
    parser.add_argument('unit', type=int, nargs='?')
    parser.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID') or 0))
    parser.add_argument('--num-tasks', type=int,
                        default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT') or 1))
    args = parser.parse_args()
    if args.command == 'run':
        run(args.sweep, args.task, args.num_tasks)
    elif args.command == 'merge':
        merge(args.sweep)
    else:
        check(args.sweep, args.unit)
