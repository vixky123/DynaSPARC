# Single-Trial P300 Analysis on NEMAR nm000231 (inspired by DynaSPARC)

This project tests the reaction-time-free ideas of **DynaSPARC** (Chen et al., 2026, *J. Neural Eng.* 23 056014) on the public EEG dataset **NEMAR nm000231** (Hoffmann et al., 2008, EPFL P300 dataset).

It answers two questions:

1. Does the P300 latency vary from trial to trial in this dataset?
2. Does an adaptive, per-trial P300 window (estimated from the EEG itself) classify Target vs NonTarget better than a fixed window?

Everything runs from one script: `p300_dynasparc_nm000231.py`.

---

## Dataset

| Item | Details |
|---|---|
| Source | NEMAR nm000231, mirrored at https://github.com/nemarDatasets/nm000231 |
| Subjects | 8 adults: S1–S4 disabled, S6–S9 able-bodied (S5 not released) |
| Task | Six images flash every 400 ms; the subject silently counts one target image |
| EEG | 32 channels, 2048 Hz, BioSemi ActiveTwo, BIDS format (.bdf + events.tsv) |
| Size | 192 recordings (8 subjects × 4 sessions × 6 runs), about 2 GB |
| Labels | Target / NonTarget flash (about 1 in 6 is a Target) |

The dataset has **no reaction times**, so DynaSPARC's RT-driven window module (DTW) cannot be reproduced here.

---

## Requirements

- Python 3.9 or newer
- Packages: `mne numpy scipy pandas matplotlib scikit-learn requests`
- About 3 GB free disk space
- No GPU needed

```bash
pip install mne numpy scipy pandas matplotlib scikit-learn requests
```

---

## How to run (Google Colab)

Run each block in its own Colab cell.

**1. Install MNE** (other packages are already in Colab)

```python
!pip install mne -q
```

Warnings about `jedi`, `decorator`, `docutils` or `sphinx` are harmless and can be ignored.

**2. Upload the script**

Click the folder icon in the left sidebar, then upload `p300_dynasparc_nm000231.py`.

**3. Get the dataset structure from GitHub**

```python
!git clone -q https://github.com/nemarDatasets/nm000231
```

The `.bdf` files in the clone are only small pointer files at this stage.

**4. Download the real EEG files**

```python
import os, glob, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://nemar.s3.us-east-2.amazonaws.com/nm000231/objects/"
files = sorted(glob.glob("nm000231/sub-*/ses-*/eeg/*_eeg.bdf"))
print(len(files), "EEG files to download")

def fetch(path):
    if os.path.getsize(path) > 10_000:          # already downloaded
        return "skip"
    key = os.path.basename(open(path).read().strip())
    expected = int(key.split("-s")[1].split("--")[0])
    tmp = path + ".part"
    with requests.get(BASE + key, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if os.path.getsize(tmp) != expected:
        raise IOError(f"incomplete download: {path}")
    os.replace(tmp, path)
    return "ok"

with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(fetch, p): p for p in files}
    for fut in as_completed(futures):
        try:
            fut.result()
        except Exception as e:
            print("FAILED:", futures[fut], "->", e)
print("Finished")
```

If any file fails, run the cell again; finished files are skipped. Check that the `.bdf` files are about 10 MB each:

```python
!ls -lh nm000231/sub-1/ses-0/eeg/ | grep bdf
```

Note: `aws s3 sync` does not work for this dataset (folder listing is not allowed and files are stored under content keys).

**5. Run the analysis**

Quick test (2 subjects, 2 sessions, about 3 minutes):

```python
!python p300_dynasparc_nm000231.py --bids-root nm000231 --subjects 1 6 --sessions 0 1 --out test_results
```

Full run (all 8 subjects, 4 sessions, about 10–20 minutes):

```python
!python p300_dynasparc_nm000231.py --bids-root nm000231 --out results
```

**6. View and download the results**

```python
from IPython.display import Image, display
import glob
for f in sorted(glob.glob("results/*.png")):
    print(f); display(Image(f))

!zip -r results.zip results
from google.colab import files
files.download("results.zip")
```

Colab deletes files when the session ends, so download `results.zip` before closing.

---

## Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--bids-root` | `nm000231` | Path to the dataset folder |
| `--out` | `results` | Folder for figures and CSV files |
| `--subjects` | `1 2 3 4 6 7 8 9` | Subjects to analyse |
| `--sessions` | all (`0 1 2 3`) | Sessions to use; at least 2 are needed for cross-validation |
| `--demo` | off | Run on simulated EEG to test the pipeline without downloading data |

---

## What the script does

1. **Preprocessing:** average reference, 0.5–15 Hz band-pass filter, resample to 128 Hz.
2. **Epoching:** −100 to 800 ms around every flash, baseline −100 to 0 ms.
3. **ERP analysis:** Target vs NonTarget waveforms at Pz and scalp maps of the difference (±50 ms around the peak).
4. **Single-trial latency:** P300 peak found in 250–600 ms on the centro-parietal average (Cz, CP1, CP2, Pz, P3, P4) using softmax and soft-argmax (DynaSPARC eqs. 26–27).
5. **Classification:** Target vs NonTarget with shrinkage LDA (equal priors), leave-one-session-out cross-validation, comparing four versions:

| Version | Channels | Time window |
|---|---|---|
| Baseline | All 32 | Fixed 0–800 ms |
| + Spatial prior | Top 8 from P300 scalp map (eqs. 17–18) | Fixed 0–800 ms |
| + Adaptive window | All 32 | Peak ± 200 ms per trial (matched filter + soft-argmax) |
| Full | Top 8 | Peak ± 200 ms per trial |

The scalp-map prior and P300 template are built from training sessions only, so no test data leaks into training.

---

## Outputs

| File | Content |
|---|---|
| `fig1_erp_target_vs_nontarget.png` | Average waveforms at Pz (all, disabled, able-bodied) |
| `fig2_p300_topography.png` | Scalp maps of Target − NonTarget at the peak |
| `fig3_latency_jitter.png` | Single-trial latency per subject and jitter by group |
| `fig4_aligned_vs_conventional.png` | Conventional vs latency-aligned averages, with NonTarget control |
| `fig5_ablation.png` | AUC and balanced accuracy for the four versions |
| `subjects/S*_erp_image.png` | Target trials of each subject sorted by latency |
| `latency_summary.csv` | Median latency and jitter (SD, IQR) per subject |
| `single_trial_latencies.csv` | Every Target trial's estimated latency |
| `classification_per_fold.csv` | Metrics for each test session |
| `classification_per_subject.csv` | Metrics averaged per subject |
| `ablation_table.csv` | Mean and SD of each metric per version |
| `config_used.json` | Settings used for the run |

---

## Main results (full run, 8 subjects)

- The P300 is clear over centro-parietal sites at about 400 ms.
- Single-trial latency varies by 61–83 ms (SD) in every subject.
- Aligning trials to their own peak also creates a P300-like peak in NonTarget trials (about 1.8 µV), so most of the apparent gain is an artefact.

| Version | AUC | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| Baseline | 0.872 | 0.801 | 0.757 |
| + Spatial prior | 0.819 | 0.748 | 0.685 |
| + Adaptive window | 0.747 | 0.689 | 0.632 |
| Full | 0.698 | 0.649 | 0.588 |

The fixed window was best for all 8 subjects. Estimating the window from the EEG alone locks onto noise, which supports DynaSPARC's use of an external guide (reaction time).

---

## Limitations

- This is not a reproduction of DynaSPARC: it uses LDA instead of the paper's CNN, and Target vs NonTarget detection instead of 3-level grading.
- The adaptive window comes from the EEG, because the dataset has no reaction times.
- Only 8 subjects, so group comparisons are not conclusive.
- Flashes every 400 ms cause overlapping responses within each epoch.
- Eye artefacts are not removed, and parameters were not tuned.

---

## References

1. Chen H et al. 2026. DynaSPARC: a dynamic single-trial P300 paradigm for assessing motor response capability in autism spectrum disorder. *J. Neural Eng.* 23 056014. https://doi.org/10.1088/1741-2552/ae9ef0
2. Hoffmann U, Vesin J-M, Ebrahimi T, Diserens K. 2008. An efficient P300-based brain–computer interface for disabled subjects. *J. Neurosci. Methods* 167 115–125. https://doi.org/10.1016/j.jneumeth.2007.03.005
3. NEMAR dataset nm000231 (v1.0.3). https://github.com/nemarDatasets/nm000231
4. Gramfort A et al. 2013. MEG and EEG data analysis with MNE-Python. *Front. Neurosci.* 7 267.
5. Pedregosa F et al. 2011. Scikit-learn: machine learning in Python. *J. Mach. Learn. Res.* 12 2825–2830.
