"""The toy-model figures of the paper, drawn from the merged sweeps in data/.

  python plot.py entropy_gap|topk_forward_kl|jsd_heatmaps|topk_jsd|all [--data DIR]

writes figures/toy_model_<figure>.pdf. Curves and markers are means over seeds, bands one standard deviation.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA = 'data'
COLOR = {'forward': 'C0', 'reverse': 'C3'}


def load(sweep, data_dir=None):
    """The units of a merged sweep file (keys '<unit>/<field>', see sweep.merge) as dicts."""
    units = {}
    with np.load(os.path.join(data_dir or DATA, f'{sweep}.npz')) as f:
        for key in f.files:
            if '/' in key:
                i, field = key.split('/')
                units.setdefault(int(i), {})[field] = f[key].item() if f[key].ndim == 0 else f[key]
    return [units[i] for i in sorted(units)]


def _select(units, **conditions):
    return [u for u in units if all(u.get(k) == v for k, v in conditions.items())]


def _seed_stats(units, x, y):
    """(x, mean over seeds of y, std over seeds of y), sorted by x."""
    units = sorted(units, key=lambda u: u[x])
    values = np.stack([np.atleast_1d(u[y]) for u in units])
    return np.array([u[x] for u in units]), values.mean(axis=1), values.std(axis=1)


def _save(fig, name):
    os.makedirs('figures', exist_ok=True)
    path = os.path.join('figures', f'toy_model_{name}.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'wrote {path}')


def entropy_gap():
    """Entropy gap against load (b), teacher perplexity (c) and JS weight (d), and entropy
    trajectories along the JS family (e)."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, sweep, x, label in [(axes[0, 0], 'context_load', 'n', 'number of contexts'),
                                (axes[0, 1], 'perplexity_gap', 'perplexity', 'teacher perplexity')]:
        units = load(sweep)
        for divergence in ('forward', 'reverse'):
            xs, mean, std = _seed_stats(_select(units, divergence=divergence), x, 'dH')
            ax.plot(xs, mean, 'o-', color=COLOR[divergence], label=f'{divergence} KL')
            ax.fill_between(xs, mean - std, mean + std, color=COLOR[divergence], alpha=0.2)
        ax.set(xscale='log', xlabel=label, ylabel='average entropy gap')
        ax.axhline(0, color='grey', lw=0.5)
        ax.legend()

    family = load('jsd_family')
    alpha, mean, std = _seed_stats(_select(family, kind='endpoint'), 'alpha', 'dH')
    axes[1, 0].errorbar(alpha, mean, yerr=std, fmt='o-', color='k')
    axes[1, 0].axhline(0, color='grey', lw=0.5)
    axes[1, 0].set(xscale='log', xlabel=r'generalized JS weight $\alpha$',
                   ylabel='average entropy gap')

    trajectories = sorted(_select(family, kind='trajectory'), key=lambda u: u['alpha'])
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(trajectories)))
    for u, color in zip(trajectories, colors):
        axes[1, 1].plot(u['steps'], u['entropy'], color=color, label=rf'$\alpha$ = {u["alpha"]:g}')
    axes[1, 1].axhline(trajectories[0]['H_T'], color='grey', ls='--', label='teacher')
    axes[1, 1].set(xscale='log', xlabel='step', ylabel='average entropy (seed 0)')
    axes[1, 1].legend()
    for ax, letter in zip(axes.flat, 'bcde'):
        ax.set_title(letter, loc='left', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'entropy_gap')


def _topk(alphas, name):
    """Student entropy under top-k distillation: rows are JS weights, columns the window (student
    or teacher top-k) and what happens to the rest of the mass (renormalized or tail bucket)."""
    units = load('topk')
    ks = sorted({u['k'] for u in units})
    colors = dict(zip(ks, plt.cm.viridis(np.linspace(0, 0.9, len(ks)))))
    columns = [(w, t) for w in ('student', 'teacher') for t in (False, True)]
    fig, axes = plt.subplots(len(alphas), 4, figsize=(12, 2.6 * len(alphas) + 0.4),
                             sharex=True, sharey=True, squeeze=False)
    for row, alpha in zip(axes, alphas):
        for ax, (window, tail) in zip(row, columns):
            for k in ks:
                if k == 1 and not tail:  # a renormalized one-token window has zero loss
                    continue
                u, = _select(units, alpha=alpha, window=window, tail=tail, k=k)
                mean, std = u['entropy'].mean(axis=0), u['entropy'].std(axis=0)
                ax.fill_between(u['steps'], mean - std, mean + std, color=colors[k], alpha=0.2)
                ax.plot(u['steps'], mean, color=colors[k], label=f'k = {k}')
            full, = _select(units, alpha=alpha, window=window, tail=tail, k=ks[-1])
            ax.axhline(np.mean(full['H_T']), color='grey', ls='--', label='teacher')
            ax.set(xscale='log', xlim=(1, 10000),
                   title=f'{window} top-k, {"tail bucket" if tail else "renormalized"}')
        row[0].set_ylabel(rf'$\alpha$ = {alpha:g}' + '\naverage student entropy')
    for ax in axes[-1]:
        ax.set_xlabel('step')
    axes[0, 1].legend()
    fig.tight_layout()
    _save(fig, name)


def topk_forward_kl():
    """Top-k distillation with the forward KL."""
    _topk([0.0], 'topk_forward_kl')


def topk_jsd():
    """Top-k distillation along the JS family."""
    _topk([0.0, 0.01, 0.1, 0.5, 1.0], 'topk_jsd')


def jsd_heatmaps():
    """Entropy gap over teacher perplexity and number of contexts, for five JS weights."""
    units = load('jsd_heatmap')
    alphas = sorted({u['alpha'] for u in units})
    ppls = np.array(sorted({u['perplexity'] for u in units}))
    loads = np.array(sorted({u['n'] for u in units}))
    grids = []
    for alpha in alphas:
        g = np.full((len(ppls), len(loads)), np.nan)
        for u in _select(units, alpha=alpha):
            g[np.searchsorted(ppls, u['perplexity']), np.searchsorted(loads, u['n'])] = np.mean(u['dH'])
        grids.append(g)
    lim = np.nanmax(np.abs(grids))
    fig, axes = plt.subplots(1, len(alphas), figsize=(15, 3.6), sharey=True)
    for ax, alpha, g in zip(axes, alphas, grids):
        mesh = ax.pcolormesh(loads, ppls, g, cmap='RdBu', vmin=-lim, vmax=lim, shading='nearest')
        ax.set(xscale='log', yscale='log', xlabel='number of contexts', title=rf'$\alpha$ = {alpha:g}')
    axes[0].set_ylabel('teacher perplexity')
    fig.colorbar(mesh, ax=axes, label='average entropy gap')
    _save(fig, 'jsd_heatmaps')


FIGURES = {'entropy_gap': entropy_gap, 'topk_forward_kl': topk_forward_kl, 'jsd_heatmaps': jsd_heatmaps, 'topk_jsd': topk_jsd}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('figure', choices=list(FIGURES) + ['all'])
    parser.add_argument('--data', default='data', help='directory of the merged sweeps')
    args = parser.parse_args()
    DATA = args.data
    for name, draw in FIGURES.items():
        if args.figure in (name, 'all'):
            draw()
