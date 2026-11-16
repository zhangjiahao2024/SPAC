
# Reproducibility

This document provides step-by-step instructions for reproducing the SPAC calibration experiments, analyses, and figures.

## Directory Structure

```text
SPAC/
├── code/                  # Source code and analysis notebooks
├── data/                  # Input trajectory dataset
├── logs/                  # Execution logs
├── work/                  # Prepared data, offline elites, and priors
└── results_spac/          # Experimental results, tables, and figures
```

The `work/` and `results_spac/` directories are generated during execution.

## Installation

Activate the Python environment and install the required dependencies:

```bash
conda activate Torch
pip install numpy scipy pandas numba pyarrow pymoo matplotlib seaborn nbformat nbconvert ipykernel
```

All commands should be executed from the root directory of the SPAC repository.

## Step 1. Data Preparation

Place the processed ARED trajectory in the `data/Cars_ARED.parquet` directory.

Run the preprocessing script:

```bash
python -u code/prepare_two_stage.py --data data/Cars_ARED.parquet --root .
```


## Step 2. Run Calibration Experiments

The experiments are organized into four phases. Complete each phase before proceeding to the next.

### Phase 1: Baseline Comparisons

This phase includes offline elite generation, prior training, and comparisons with six baseline configurations.

```bash
python -u code/run_all_spac.py --phase 1 --root . --prep work/prep.pkl
```

**Expected completion message:** `PHASE 1 COMPLETE`

### Phase 2: Prior and Descriptor Ablation

```bash
python -u code/run_all_spac.py --phase 2 --root . --prep work/prep.pkl
```

**Expected completion message:** `PHASE 2 COMPLETE`

### Phase 3: Spectral Component Ablation

```bash
python -u code/run_all_spac.py --phase 3 --root . --prep work/prep.pkl
```

**Expected completion message:** `PHASE 3 COMPLETE`

### Phase 4: Hyperparameter Sensitivity Analysis

```bash
python -u code/run_all_spac.py --phase 4 --root . --prep work/prep.pkl
```

**Expected completion message:** `PHASE 4 COMPLETE`

This phase evaluates 35 configurations, including one reference configuration, 26 single-parameter variations, and eight interaction configurations. Parameter settings are defined in `run_spac.py` under `SENS`.

## Step 3. Analyze Experimental Results

After completing the experiments, generate the analysis outputs for Phases 1–3:

```bash
python -u code/analyze_spac.py --phase 1 --root . --prep work/prep.pkl
python -u code/analyze_spac.py --phase 2 --root . --prep work/prep.pkl
python -u code/analyze_spac.py --phase 3 --root . --prep work/prep.pkl
```

Then run the section-specific analyses:

```bash
python -u code/analyze_sec_6_1.py --root . --prep work/prep.pkl --rebuild-cache
python -u code/analyze_sec_6_2.py --root . --prep work/prep.pkl
python -u code/analyze_sec_6_3.py --root . --prep work/prep.pkl
python -u code/analyze_sec_6_4.py --root . --prep work/prep.pkl
python -u code/check_acceleration_plausibility.py --root . --prep work/prep.pkl
```

## Outputs

The workflow generates two main directories:

- `work/`: Preprocessed trajectories, offline elite solutions, and learned priors.
- `results_spac/`: Calibration results, statistical analyses, tables, and figures.

