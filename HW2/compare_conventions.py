#!/usr/bin/env python3
"""
================================================================================
compare_conventions.py: Direct Quantitative & Visual Comparison of Conventions
================================================================================

Compares:
  - Convention 1: Strict global distance shells (Shell 1: r_1 ≈ 0.901a inter; Shell 2: r_2 = a intra-x; y-axial excluded)
  - Convention 2: Per-sublattice/per-block shells (Intra NN: a and b; Intra NNN: sqrt(a^2+b^2); Inter NN: r_1; Inter NNN: 8 extended)

Generates:
  1. Detailed comparison metrics across k-space for a in {5.0, 7.0, 9.0} and b = 1.5*a
  2. Overlay figures showing Convention 1 (dashed) vs Convention 2 (solid)
  3. Difference spectra |E_conv2(k) - E_conv1(k)|
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import convention_1 as c1
import convention_2 as c2

def run_comparison():
    a_values = [5.0, 7.0, 9.0]
    modes = ['Full', 'No Overlap', 'NN only']
    num_seg = 150

    results_table = []

    print("=" * 80)
    print("CONVENTION 1 vs CONVENTION 2: QUANTITATIVE COMPARISON (b = 1.5 * a)")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Numerical Comparison Loop
    # --------------------------------------------------------------------------
    data_store = {}

    for a_val in a_values:
        b_val = 1.5 * a_val
        k_path, dists, ticks, labels = c1.build_rectangular_bz_path(a_val, b_val, num_seg=num_seg)
        p1 = c1.get_convention1_params(a_val, b_val)
        p2 = c2.get_convention2_params(a_val, b_val)

        data_store[a_val] = {'dists': dists, 'ticks': ticks, 'labels': labels, 'modes': {}}

        print(f"\n>>> Lattice parameter a = {a_val:.1f} d_0, b = {b_val:.1f} d_0:")
        print(f"{'Mode':<15} | {'Max Diff (t_0)':<16} | {'Mean Diff (t_0)':<16} | {'RMS Diff (t_0)':<16} | {'Max Diff Point'}")
        print("-" * 80)

        for mode in modes:
            e1 = c1.solve_rectangular_path_conv1(a_val, b_val, k_path, mode=mode)
            e2 = c2.solve_rectangular_path_conv2(a_val, b_val, k_path, mode=mode)

            diff = np.abs(e2 - e1)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            rms_diff = np.sqrt(np.mean(diff**2))

            # Find k-point of max diff
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            max_k_dist = dists[max_idx[0]]
            max_band = max_idx[1] + 1

            # Check nearest high symmetry label
            closest_tick = np.argmin(np.abs(np.array(ticks) - max_k_dist))
            label_near = labels[closest_tick]

            data_store[a_val]['modes'][mode] = {
                'e1': e1, 'e2': e2, 'diff': diff,
                'max_diff': max_diff, 'mean_diff': mean_diff, 'rms_diff': rms_diff,
                'max_idx': max_idx, 'label_near': label_near
            }

            results_table.append({
                'a': a_val, 'b': b_val, 'mode': mode,
                'max_diff': max_diff, 'mean_diff': mean_diff, 'rms_diff': rms_diff,
                'label_near': label_near, 'band': max_band
            })

            print(f"{mode:<15} | {max_diff:<16.5f} | {mean_diff:<16.5f} | {rms_diff:<16.5f} | Near {label_near} (Band {max_band})")

    # --------------------------------------------------------------------------
    # 2. Plotting: Comprehensive 3x3 Overlay Figure
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharey='row', sharex='col')
    plt.subplots_adjust(hspace=0.25, wspace=0.15)

    band_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for row_idx, mode in enumerate(modes):
        for col_idx, a_val in enumerate(a_values):
            ax = axes[row_idx, col_idx]
            d = data_store[a_val]
            dists = d['dists']
            ticks = d['ticks']
            labels = d['labels']
            res = d['modes'][mode]

            e1 = res['e1']
            e2 = res['e2']

            # Plot Convention 2 (Solid lines)
            for b in range(3):
                lbl2 = f"Conv 2 (Per-sublattice, Band {b+1})" if (row_idx == 0 and col_idx == 0) else None
                ax.plot(dists, e2[:, b], color=band_colors[b], linestyle='-', linewidth=2.0, label=lbl2)

            # Plot Convention 1 (Dashed lines, darkened)
            for b in range(3):
                lbl1 = f"Conv 1 (Global shells, Band {b+1})" if (row_idx == 0 and col_idx == 0) else None
                ax.plot(dists, e1[:, b], color='black' if b==0 else ('#8c564b' if b==1 else '#7f7f7f'),
                        linestyle='--', linewidth=1.8, alpha=0.85, label=lbl1)

            # Format axis
            for tick in ticks:
                ax.axvline(tick, color='#cccccc', linestyle=':', linewidth=0.8)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
            ax.set_xlim(dists[0], dists[-1])
            ax.grid(True, linestyle=':', alpha=0.4)

            # Titles and labels
            if row_idx == 0:
                ax.set_title(rf"$a = {a_val:.1f}\,d_0$, $b = {1.5*a_val:.1f}\,d_0$", fontsize=13, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(rf"{mode}" + "\n" + r"$E(\mathbf{k}) / t_0$", fontsize=12, fontweight='bold')

            # Annotation of max difference
            ax.text(0.04, 0.92, f"Max $\\Delta E = {res['max_diff']:.4f}\\,t_0$\nMean $\\Delta E = {res['mean_diff']:.4f}\\,t_0$",
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#aaaaaa'))

    axes[0, 0].legend(loc='lower left', fontsize=9, framealpha=0.9)

    fig.suptitle("Crystal B ($b = 1.5a$): Convention 1 (Global Shells, Dashed) vs Convention 2 (Per-Sublattice Shells, Solid)",
                 fontsize=15, fontweight='bold', y=0.98)

    target_dirs = [
        os.path.join(os.path.dirname(__file__), 'plots'),
        os.path.join(os.path.dirname(__file__), 'Writeup', 'figures')
    ]
    for t_dir in target_dirs:
        os.makedirs(t_dir, exist_ok=True)
        out_path = os.path.join(t_dir, 'convention_comparison_overlay.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved overlay figure: {out_path}")

    plt.close(fig)

    # --------------------------------------------------------------------------
    # 3. Dedicated a = 7.0 Comparison Focus Figure
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    plt.subplots_adjust(wspace=0.15)

    a_focus = 7.0
    d = data_store[a_focus]

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        res = d['modes'][mode]
        e1, e2 = res['e1'], res['e2']

        for b in range(3):
            ax.plot(d['dists'], e2[:, b], color=band_colors[b], linestyle='-', linewidth=2.2,
                    label=f"Conv 2 (Band {b+1})" if idx == 0 else None)
            ax.plot(d['dists'], e1[:, b], color='black' if b==0 else ('#555555' if b==1 else '#888888'),
                    linestyle='--', linewidth=1.8,
                    label=f"Conv 1 (Band {b+1})" if idx == 0 else None)

        for tick in d['ticks']:
            ax.axvline(tick, color='#cccccc', linestyle=':', linewidth=0.8)
        ax.set_xticks(d['ticks'])
        ax.set_xticklabels(d['labels'], fontsize=11, fontweight='bold')
        ax.set_xlim(d['dists'][0], d['dists'][-1])
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_title(f"Mode: {mode}\n(Max $\\Delta E = {res['max_diff']:.4f}\\,t_0$)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Wavevector $\\mathbf{k}$ Path", fontsize=11)

    axes[0].set_ylabel("$E(\\mathbf{k}) / t_0$", fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower left', fontsize=9, framealpha=0.9)
    fig.suptitle(f"Crystal B at $a = 7.0\\,d_0, b = 10.5\\,d_0$: Convention 1 vs Convention 2 Across Approximation Modes",
                 fontsize=14, fontweight='bold', y=1.02)

    for t_dir in target_dirs:
        out_path = os.path.join(t_dir, 'convention_comparison_a7_focus.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved focus figure: {out_path}")

    plt.close(fig)

    print("\n[COMPLETE] Comparison figures and tables successfully generated.")

if __name__ == '__main__':
    run_comparison()
