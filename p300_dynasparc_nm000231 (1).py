#!/usr/bin/env python3
"""
Single-trial P300 analysis of NEMAR dataset nm000231
(EPFL P300 dataset, Hoffmann et al. 2008, J. Neurosci. Methods 167:115-125)
with ideas borrowed from DynaSPARC (Chen et al. 2026, J. Neural Eng. 23 056014).

What this script does
---------------------
1. Loads the BIDS data (BDF files + events.tsv) for every subject / session / run.
2. Preprocesses: average reference, 0.5-15 Hz band-pass, resample to 128 Hz,
   epochs -100..800 ms around every flash, baseline-corrected.
3. Classic ERP analysis: Target vs NonTarget waveforms and P300 scalp maps.
4. Single-trial P300 latency estimation (hard argmax and the paper's
   softmax / soft-argmax, eqs. 26-27) -> latency jitter per subject,
   disabled vs able-bodied comparison, ERP images, latency-aligned averages.
5. DynaSPARC-style ablation on Target vs NonTarget classification
   (shrinkage LDA, leave-one-session-out CV):
       Baseline      : fixed 0-800 ms window, all 32 channels
       + Spatial     : topography prior (eqs. 17-18) -> top-K weighted channels
       + Temporal    : matched-filter + soft-argmax peak -> trial-adaptive window
       Full          : spatial + temporal
   NOTE: this dataset has NO reaction times, so DynaSPARC's RT-driven DTW
   module cannot be reproduced. The "temporal" branch here follows the
   RT-free part of CSTA (latency estimation from the EEG itself).

Usage
-----
    # 1) download the data (about 2-3 GB):
    #       nemar dataset download nm000231
    # 2) run:
    python p300_dynasparc_nm000231.py --bids-root ./nm000231 --out results
    # quick test on 2 subjects, 2 sessions:
    python p300_dynasparc_nm000231.py --bids-root ./nm000231 --subjects 1 6 --sessions 0 1
    # no data yet? check the pipeline end-to-end with simulated EEG:
    python p300_dynasparc_nm000231.py --demo

Requirements: pip install mne numpy scipy pandas matplotlib scikit-learn
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

mne.set_log_level("ERROR")

# ----------------------------------------------------------------------------
# Dataset facts (from the dataset README / original paper)
# ----------------------------------------------------------------------------
# Subjects 1-4: disabled (cerebral palsy, MS, late-stage ALS, brain + C4 spinal
# cord injury). Subjects 6-9: able-bodied. Subject 5 was not released.
DISABLED = {1, 2, 3, 4}
ABLE = {6, 7, 8, 9}
ALL_SUBJECTS = sorted(DISABLED | ABLE)

CH_NAMES = ["Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5",
            "P7", "P3", "Pz", "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8",
            "CP6", "CP2", "C4", "T8", "FC6", "FC2", "F4", "F8", "AF4", "Fp2",
            "Fz", "Cz"]
ROI = ["Cz", "CP1", "CP2", "Pz", "P3", "P4"]   # centro-parietal P300 region

CFG = dict(
    l_freq=0.5, h_freq=15.0,     # band-pass (Hz)
    sfreq=128.0,                 # resampling rate (Hz)
    tmin=-0.1, tmax=0.8,         # epoch (s); SOA is 400 ms so epochs overlap
    baseline=(-0.1, 0.0),
    search=(0.25, 0.60),         # P300 peak search range (s)
    ref_window=(0.30, 0.60),     # topography-prior window, as in paper eq. 17
    template=(0.20, 0.60),       # matched-filter template span (s)
    half_win=0.20,               # adaptive window = peak +/- half_win (s)
    fixed_window=(0.0, 0.8),     # fixed-window baseline (s)
    decim=4,                     # 128 Hz -> 32 Hz for classifier features
    top_k=8,                     # channels kept by the spatial prior
    tau=0.5,                     # softmax temperature (paper eq. 26)
    eps=1e-3,                    # clip value in paper eq. 18
)

GROUP_COLORS = {"disabled": "#d1495b", "able-bodied": "#2e86ab"}


def group_of(subject):
    return "disabled" if subject in DISABLED else "able-bodied"


# ----------------------------------------------------------------------------
# 1. Loading and preprocessing
# ----------------------------------------------------------------------------
def find_runs(bids_root, subject, sessions=None):
    pattern = os.path.join(bids_root, f"sub-{subject}", "ses-*", "eeg",
                           f"sub-{subject}_ses-*_task-p300_run-*_eeg.bdf")
    runs = []
    for bdf in sorted(glob.glob(pattern)):
        ses = int(bdf.split("_ses-")[1].split("_")[0])
        if sessions is not None and ses not in sessions:
            continue
        runs.append((ses, bdf, bdf.replace("_eeg.bdf", "_events.tsv")))
    return runs


def read_bdf(bdf_path):
    real_path = os.path.realpath(bdf_path)
    if not os.path.exists(real_path) or os.path.getsize(real_path) < 10_000:
        raise FileNotFoundError(
            f"{bdf_path} is only an annex pointer, the EEG itself is not downloaded.\n"
            "Run `nemar dataset download nm000231` (or `datalad get .` inside the "
            "dataset folder) and try again.")
    raw = mne.io.read_raw_bdf(bdf_path, preload=True)
    missing = [c for c in CH_NAMES if c not in raw.ch_names]
    if missing:
        raise ValueError(f"Channels missing in {bdf_path}: {missing}")
    return raw.pick(CH_NAMES)


def preprocess_and_epoch(raw, events_df, cfg):
    """Raw (Volts) + BIDS events table -> X (trials, ch, time) in microvolts."""
    raw.set_montage("standard_1020", on_missing="ignore")
    raw.set_eeg_reference("average", projection=False)
    raw.filter(cfg["l_freq"], cfg["h_freq"], fir_design="firwin")
    raw.resample(cfg["sfreq"])

    codes = {"NonTarget": 1, "Target": 2}
    ev = events_df[events_df["trial_type"].isin(codes)]
    samples = np.round(ev["onset"].to_numpy() * raw.info["sfreq"]).astype(int)
    events = np.c_[samples + raw.first_samp, np.zeros(len(ev), int),
                   ev["trial_type"].map(codes).to_numpy()]

    epochs = mne.Epochs(raw, events, event_id=codes, tmin=cfg["tmin"],
                        tmax=cfg["tmax"], baseline=cfg["baseline"],
                        preload=True)
    X = epochs.get_data(copy=False) * 1e6               # V -> uV
    y = (epochs.events[:, 2] == 2).astype(int)          # 1 = Target
    return X, y, epochs.times


def load_subject(subject, cfg, bids_root=None, sessions=None, demo_rng=None):
    Xs, ys, ss = [], [], []
    if demo_rng is not None:
        runs = [(ses, run) for ses in (sessions or [0, 1, 2, 3]) for run in range(6)]
    else:
        runs = find_runs(bids_root, subject, sessions)
        if not runs:
            raise FileNotFoundError(f"No runs found for sub-{subject} in {bids_root}")
    for item in runs:
        if demo_rng is not None:
            ses, run = item
            raw, events_df = make_synthetic_run(subject, run, demo_rng)
        else:
            ses, bdf, tsv = item
            raw, events_df = read_bdf(bdf), pd.read_csv(tsv, sep="\t")
        X, y, times = preprocess_and_epoch(raw, events_df, cfg)
        Xs.append(X), ys.append(y), ss.append(np.full(len(y), ses))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss), times


# ----------------------------------------------------------------------------
# Synthetic data (only for --demo): same layout, timing and channels as the
# real dataset, with a jittered P300 injected on Target flashes.
# ----------------------------------------------------------------------------
def make_synthetic_run(subject, run, rng, sfreq=512.0, duration=55.0):
    info = mne.create_info(CH_NAMES, sfreq, "eeg")
    info.set_montage("standard_1020")
    pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(CH_NAMES))])
    pz = pos[CH_NAMES.index("Pz")]
    spatial = np.exp(-np.sum((pos - pz) ** 2, 1) / (2 * 0.04 ** 2))

    n = int(duration * sfreq)
    t = np.arange(n) / sfreq
    noise = np.cumsum(rng.standard_normal((len(CH_NAMES), n)), 1)   # 1/f-ish
    noise = mne.filter.filter_data(noise, sfreq, 0.3, 40, verbose=False)
    data = noise / noise.std() * 6.0 + rng.standard_normal(noise.shape) * 2.0

    disabled = subject in DISABLED
    lat_mu, lat_sd = (0.42, 0.08) if disabled else (0.36, 0.035)
    amp = 5.0 if disabled else 7.0
    onsets = np.arange(0.4, duration - 1.0, 0.4)
    order = np.concatenate([rng.permutation(6) for _ in range(len(onsets) // 6 + 1)])
    target_img = run % 6
    rows = []
    for k, onset in enumerate(onsets):
        is_t = order[k] == target_img
        n1 = -2.0 * np.exp(-((t - onset - 0.17) ** 2) / (2 * 0.02 ** 2))
        data += n1 * 0.5
        if is_t:
            lat = onset + np.clip(rng.normal(lat_mu, lat_sd), 0.26, 0.6)
            p3 = amp * np.exp(-((t - lat) ** 2) / (2 * 0.06 ** 2))
            data += spatial[:, None] * p3[None, :]
        rows.append(dict(onset=onset, duration=0.1,
                         trial_type="Target" if is_t else "NonTarget"))
    raw = mne.io.RawArray(data * 1e-6, info)
    return raw, pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2. Single-trial latency estimation (paper eqs. 24-27)
# ----------------------------------------------------------------------------
def roi_signal(X):
    idx = [CH_NAMES.index(c) for c in ROI]
    return X[:, idx, :].mean(axis=1)                      # eq. 24: channel fusion


def softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def peak_latency(roi, times, cfg, template=None):
    """Return (soft-argmax latency, hard-argmax latency) in seconds per trial.

    template=None : intensity r(t) = ROI amplitude itself
    template=h    : r(t) = matched-filter output (paper eq. 25, Conv1D)
    """
    s0, s1 = cfg["search"]
    if template is None:
        mask = (times >= s0) & (times <= s1)
        r, cand = roi[:, mask], times[mask]
    else:
        L = len(template)
        win = sliding_window_view(roi, L, axis=1)        # (n, T-L+1, L)
        centers = times[np.arange(win.shape[1]) + L // 2]
        mask = (centers >= s0) & (centers <= s1)
        r, cand = win[:, mask, :] @ template, centers[mask]
    z = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-12)
    p = softmax(z / cfg["tau"])                          # eq. 26
    t_soft = p @ cand                                    # eq. 27 soft-argmax
    t_hard = cand[np.argmax(r, axis=1)]
    return t_soft, t_hard


def aligned_average(roi, times, lat, half=0.25):
    """Average ROI waveform with every trial re-centred on its own peak."""
    dt = times[1] - times[0]
    h = int(round(half / dt))
    idx = np.clip(np.searchsorted(times, lat), h, len(times) - h - 1)
    segs = np.stack([roi[i, c - h:c + h + 1] for i, c in enumerate(idx)])
    return np.arange(-h, h + 1) * dt, segs.mean(0)


# ----------------------------------------------------------------------------
# 3. Feature extraction for the ablation (fixed vs DynaSPARC-like)
# ----------------------------------------------------------------------------
def topography_prior(X_target, times, cfg):
    """Paper eqs. 17-18: mean target ERP in 300-600 ms, min-max normalised."""
    m = (times >= cfg["ref_window"][0]) & (times <= cfg["ref_window"][1])
    a = X_target[:, :, m].mean(axis=(0, 2))
    w0 = (a - a.min()) / (a.max() - a.min() + cfg["eps"])
    return np.clip(w0, cfg["eps"], 1 - cfg["eps"])


def matched_template(X_target, times, cfg):
    m = (times >= cfg["template"][0]) & (times <= cfg["template"][1])
    h = roi_signal(X_target)[:, m].mean(0)
    h = h - h.mean()
    return h / (np.linalg.norm(h) + 1e-12)


def make_features(X, times, cfg, fold, use_spatial, use_temporal):
    ch = np.arange(len(CH_NAMES))
    w = np.ones(len(ch))
    if use_spatial:
        order = np.argsort(fold["w0"])[::-1][:cfg["top_k"]]
        ch, w = order, fold["w0"][order]
    Xc = X[:, ch, :] * w[None, :, None]                 # eq. 22 spatial gating

    if use_temporal:
        lat, _ = peak_latency(roi_signal(X), times, cfg, fold["template"])
        dt = times[1] - times[0]
        h = int(round(cfg["half_win"] / dt))
        c = np.clip(np.searchsorted(times, lat), h, len(times) - h - 1)
        Xw = np.stack([Xc[i, :, ci - h:ci + h + 1] for i, ci in enumerate(c)])
    else:
        m = (times >= cfg["fixed_window"][0]) & (times <= cfg["fixed_window"][1])
        Xw = Xc[:, :, m]
    return Xw[:, :, ::cfg["decim"]].reshape(len(X), -1)


CONFIGS = {
    "Baseline (fixed)": (False, False),
    "+ Spatial prior": (True, False),
    "+ Adaptive window": (False, True),
    "Full (spatial+temporal)": (True, True),
}


def evaluate_subject(X, y, sess, times, cfg):
    rows = []
    for test_ses in np.unique(sess):
        tr, te = sess != test_ses, sess == test_ses
        Xt = X[tr][y[tr] == 1]
        fold = dict(w0=topography_prior(Xt, times, cfg),        # training only
                    template=matched_template(Xt, times, cfg))  # training only
        for name, (sp, tp) in CONFIGS.items():
            Ftr = make_features(X[tr], times, cfg, fold, sp, tp)
            Fte = make_features(X[te], times, cfg, fold, sp, tp)
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                             priors=[0.5, 0.5]).fit(Ftr, y[tr])
            score, pred = clf.decision_function(Fte), clf.predict(Fte)
            rows.append(dict(
                config=name, test_session=int(test_ses),
                auc=roc_auc_score(y[te], score),
                balanced_acc=balanced_accuracy_score(y[te], pred),
                macro_f1=f1_score(y[te], pred, average="macro"),
                precision_target=precision_score(y[te], pred, zero_division=0),
                recall_target=recall_score(y[te], pred)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. Figures
# ----------------------------------------------------------------------------
def fig_erp(store, times, out):
    pz = CH_NAMES.index("Pz")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    groups = [("All subjects", list(store)),
              ("Disabled", [s for s in store if s in DISABLED]),
              ("Able-bodied", [s for s in store if s in ABLE])]
    for ax, (title, subs) in zip(axes, groups):
        if not subs:
            ax.set_visible(False)
            continue
        for key, color, label in [("erp_t", "#d1495b", "Target"),
                                  ("erp_nt", "#555555", "NonTarget")]:
            w = np.stack([store[s][key][pz] for s in subs])
            m, se = w.mean(0), w.std(0) / np.sqrt(len(subs))
            ax.plot(times * 1000, m, color=color, lw=2, label=label)
            ax.fill_between(times * 1000, m - se, m + se, color=color, alpha=0.2)
        ax.axvspan(300, 600, color="gold", alpha=0.12)
        ax.axhline(0, color="k", lw=0.5), ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"{title} (n={len(subs)}) - Pz")
        ax.set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_erp_target_vs_nontarget.png"), dpi=150)
    plt.close(fig)


def fig_topomaps(store, times, cfg, out):
    info = mne.create_info(CH_NAMES, cfg["sfreq"], "eeg")
    info.set_montage("standard_1020")
    pz = CH_NAMES.index("Pz")
    groups = [("All", list(store)),
              ("Disabled", [s for s in store if s in DISABLED]),
              ("Able-bodied", [s for s in store if s in ABLE])]
    groups = [g for g in groups if g[1]]
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 4))
    axes = np.atleast_1d(axes)
    for ax, (title, subs) in zip(axes, groups):
        diff = np.stack([store[s]["erp_t"] - store[s]["erp_nt"] for s in subs]).mean(0)
        m = (times >= cfg["search"][0]) & (times <= cfg["search"][1])
        t_pk = times[m][np.argmax(diff[pz, m])]
        w = (times >= t_pk - 0.05) & (times <= t_pk + 0.05)   # +/-50 ms, as in paper
        im, _ = mne.viz.plot_topomap(diff[:, w].mean(1), info, axes=ax,
                                     show=False, cmap="RdBu_r")
        ax.set_title(f"{title}\nTarget-NonTarget @ {t_pk*1000:.0f} ms")
        plt.colorbar(im, ax=ax, fraction=0.046, label="uV")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_p300_topography.png"), dpi=150)
    plt.close(fig)


def fig_latency(lat_df, out):
    subs = sorted(lat_df.subject.unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5),
                             gridspec_kw=dict(width_ratios=[3, 1.3]))
    data = [lat_df[lat_df.subject == s].latency_soft_ms for s in subs]
    parts = axes[0].violinplot(data, showmedians=True)
    for body, s in zip(parts["bodies"], subs):
        body.set_facecolor(GROUP_COLORS[group_of(s)])
        body.set_alpha(0.6)
    axes[0].set_xticks(range(1, len(subs) + 1), [f"S{s}" for s in subs])
    axes[0].set_ylabel("Single-trial P300 latency (ms)")
    axes[0].set_title("Target-trial peak latency (soft-argmax) per subject")
    for g, c in GROUP_COLORS.items():
        axes[0].scatter([], [], color=c, label=g)
    axes[0].legend()

    jit = lat_df.groupby("subject").latency_soft_ms.std()
    for g, c in GROUP_COLORS.items():
        v = [jit[s] for s in subs if group_of(s) == g]
        if v:
            axes[1].bar(g, np.mean(v), color=c, alpha=0.6)
            axes[1].scatter([g] * len(v), v, color="k", zorder=3)
    axes[1].set_ylabel("Latency jitter = SD (ms)")
    axes[1].set_title("Latency jitter by group")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_latency_jitter.png"), dpi=150)
    plt.close(fig)


def fig_erp_image(subject, roi, y, times, lat, out):
    """Classic ERP image: Target trials sorted by their estimated latency."""
    rt = roi[y == 1]
    order = np.argsort(lat)
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1]))
    v = np.percentile(np.abs(rt), 95)
    axes[0].imshow(rt[order], aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                   extent=[times[0] * 1000, times[-1] * 1000, len(rt), 0])
    axes[0].plot(lat[order] * 1000, np.arange(len(rt)) + 0.5, "k", lw=1)
    axes[0].set_ylabel("Target trials (sorted by latency)")
    axes[0].set_title(f"S{subject} ({group_of(subject)}) - ROI ERP image")
    axes[1].plot(times * 1000, rt.mean(0), color="#d1495b")
    axes[1].set_xlabel("Time (ms)"), axes[1].set_ylabel("uV")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "subjects", f"S{subject}_erp_image.png"), dpi=130)
    plt.close(fig)


def fig_aligned(store, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (key, title) in zip(axes, [("t", "Target trials"),
                                       ("nt", "NonTarget trials (control)")]):
        rel = next(iter(store.values()))["rel_t"] * 1000
        conv = np.stack([store[s][f"conv_{key}"] for s in store]).mean(0)
        alig = np.stack([store[s][f"alig_{key}"] for s in store]).mean(0)
        ax.plot(rel, conv, color="#555555", lw=2, label="Conventional average")
        ax.plot(rel, alig, color="#d1495b", lw=2, label="Latency-aligned average")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(title), ax.set_xlabel("Time relative to peak (ms)")
    axes[0].set_ylabel("ROI amplitude (uV)")
    axes[0].legend()
    fig.suptitle("Latency jitter smears the averaged P300 "
                 "(alignment inflates noise too - see control)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_aligned_vs_conventional.png"), dpi=150)
    plt.close(fig)


def fig_ablation(res_subj, out):
    names = list(CONFIGS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric, label in [(axes[0], "auc", "ROC AUC"),
                              (axes[1], "balanced_acc", "Balanced accuracy")]:
        means = [res_subj[res_subj.config == n][metric].mean() for n in names]
        ax.bar(range(len(names)), means, color="#cccccc", edgecolor="k")
        for i, n in enumerate(names):
            d = res_subj[res_subj.config == n]
            for _, r in d.iterrows():
                ax.scatter(i + np.random.uniform(-0.15, 0.15), r[metric],
                           color=GROUP_COLORS[group_of(r.subject)], s=25, zorder=3)
        ax.set_xticks(range(len(names)), names, rotation=15)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel(label)
        ax.set_title(f"{label}: Target vs NonTarget (leave-one-session-out)")
    for g, c in GROUP_COLORS.items():
        axes[0].scatter([], [], color=c, label=g)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_ablation.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bids-root", default="nm000231")
    ap.add_argument("--out", default="results")
    ap.add_argument("--subjects", type=int, nargs="*", default=ALL_SUBJECTS)
    ap.add_argument("--sessions", type=int, nargs="*", default=None,
                    help="e.g. --sessions 0 1 (needs >= 2 for cross-validation)")
    ap.add_argument("--demo", action="store_true",
                    help="run on simulated EEG to test the pipeline")
    args = ap.parse_args()

    out = args.out
    os.makedirs(os.path.join(out, "subjects"), exist_ok=True)
    rng = np.random.default_rng(0) if args.demo else None
    cfg = CFG

    store, lat_rows, res_rows = {}, [], []
    for s in args.subjects:
        print(f"\n=== Subject {s} ({group_of(s)}) ===")
        X, y, sess, times = load_subject(s, cfg, args.bids_root,
                                         args.sessions, demo_rng=rng)
        print(f"  {len(y)} epochs: {y.sum()} Target / {(y == 0).sum()} NonTarget, "
              f"{len(np.unique(sess))} sessions")

        roi = roi_signal(X)
        lat_t_soft, lat_t_hard = peak_latency(roi[y == 1], times, cfg)
        lat_nt_soft, _ = peak_latency(roi[y == 0], times, cfg)
        med = np.median(lat_t_soft)
        rel, alig_t = aligned_average(roi[y == 1], times, lat_t_soft)
        _, conv_t = aligned_average(roi[y == 1], times, np.full(len(lat_t_soft), med))
        _, alig_nt = aligned_average(roi[y == 0], times, lat_nt_soft)
        _, conv_nt = aligned_average(roi[y == 0], times, np.full(len(lat_nt_soft), med))
        store[s] = dict(erp_t=X[y == 1].mean(0), erp_nt=X[y == 0].mean(0),
                        rel_t=rel, alig_t=alig_t, conv_t=conv_t,
                        alig_nt=alig_nt, conv_nt=conv_nt)
        lat_rows += [dict(subject=s, group=group_of(s), latency_soft_ms=a * 1000,
                          latency_hard_ms=b * 1000)
                     for a, b in zip(lat_t_soft, lat_t_hard)]
        fig_erp_image(s, roi, y, times, lat_t_soft, out)
        print(f"  P300 latency median {med*1000:.0f} ms, "
              f"jitter (SD) {lat_t_soft.std()*1000:.0f} ms")

        if len(np.unique(sess)) >= 2:
            r = evaluate_subject(X, y, sess, times, cfg)
            r.insert(0, "subject", s)
            res_rows.append(r)
            print(r.groupby("config")[["auc", "balanced_acc"]].mean().round(3)
                  .to_string().replace("\n", "\n  "))
        del X

    lat_df = pd.DataFrame(lat_rows)
    lat_df.to_csv(os.path.join(out, "single_trial_latencies.csv"), index=False)
    lat_sum = lat_df.groupby(["subject", "group"]).agg(
        n_target=("latency_soft_ms", "size"),
        median_latency_ms=("latency_soft_ms", "median"),
        jitter_sd_ms=("latency_soft_ms", "std"),
        jitter_iqr_ms=("latency_soft_ms", lambda v: v.quantile(.75) - v.quantile(.25)),
    ).reset_index()
    lat_sum.to_csv(os.path.join(out, "latency_summary.csv"), index=False)

    fig_erp(store, times, out)
    fig_topomaps(store, times, cfg, out)
    fig_latency(lat_df, out)
    fig_aligned(store, out)

    print("\n=== Latency summary ===")
    print(lat_sum.round(1).to_string(index=False))

    if res_rows:
        res = pd.concat(res_rows)
        res.to_csv(os.path.join(out, "classification_per_fold.csv"), index=False)
        res_subj = res.groupby(["subject", "config"]).mean(numeric_only=True) \
                      .drop(columns="test_session").reset_index()
        res_subj.to_csv(os.path.join(out, "classification_per_subject.csv"), index=False)
        fig_ablation(res_subj, out)
        table = res_subj.groupby("config").agg(["mean", "std"]).drop(columns="subject")
        table = table.reindex(list(CONFIGS))
        table.to_csv(os.path.join(out, "ablation_table.csv"))
        print("\n=== Ablation (mean over subjects) ===")
        print(res_subj.groupby("config").mean(numeric_only=True)
              .drop(columns="subject").reindex(list(CONFIGS)).round(3).to_string())

    with open(os.path.join(out, "config_used.json"), "w") as f:
        json.dump(dict(cfg=cfg, subjects=args.subjects, sessions=args.sessions,
                       demo=args.demo), f, indent=2, default=str)
    print(f"\nAll outputs written to: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
