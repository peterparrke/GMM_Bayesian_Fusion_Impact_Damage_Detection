# Toward Robust Detection of Interfacial Debonding in CFRP-Strengthened Concrete

This repository provides the code associated with the paper:

**Toward Robust Detection of Interfacial Debonding in CFRP-Strengthened Concrete: A GMM-Bayesian Fusion Approach**

The code implements a multi-sensor impact-testing workflow for detecting
interfacial debonding between carbon fiber-reinforced polymer (CFRP) and
concrete. Temporal response signals from heterogeneous sensors are converted
into feature maps, modeled with bimodal Gaussian mixture models (GMMs), and
then fused with a Bayesian decision framework for debonding localization.

Keywords: CFRP-reinforced concrete; interfacial debonding identification;
multi-sensor impact testing; Gaussian Mixture Model; Bayesian fusion.

Packaged date: 2026-06-12

## Contents

| File | Purpose |
|---|---|
| `article_gmm_bayes_pipeline.py` | Main executable pipeline. |
| `requirements.txt` | Python package dependencies. |
| `LICENSE` | MIT License. |

## Installation

Python 3.10 or newer is recommended because the script uses modern type syntax.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Minimal Usage

The experiment contains four testing faces. The expected dataset folders are
named:

```text
<DATA_ROOT>/
  pdd-1/
  pdd-2/
  pdd-3/
  pdd-4/
```

Each folder should contain TXT signal files for a 17 x 17 measurement grid. The
selected channel suffixes and modality order can be provided through the command
line, as shown below.

```powershell
python .\article_gmm_bayes_pipeline.py `
  --data-root <DATA_ROOT> `
  --output-root <OUTPUT_ROOT> `
  --datasets pdd-1 pdd-2 pdd-3 pdd-4 `
  --prefix pdd `
  --channel-suffixes 7 8 9 10 `
  --channel-modalities hammer acc highfre lowfre `
  --gt-mat none
```

For a single-dataset check:

```powershell
python .\article_gmm_bayes_pipeline.py `
  --data-root <DATA_ROOT> `
  --output-root <OUTPUT_ROOT> `
  --datasets pdd-3 `
  --prefix pdd `
  --channel-suffixes 7 8 9 10 `
  --channel-modalities hammer acc highfre lowfre `
  --gt-mat none
```

The valid modality labels are:

```text
hammer acc highfre lowfre
```

Ground-truth metrics are optional. If a ground-truth file is available, pass a
MAT file containing variable `A`:

```powershell
--gt-mat <GROUND_TRUTH_MAT>
```

Use `--gt-mat none` when metric calculation is not required.

## Processing Workflow

1. Read multi-sensor impact-response signals from the four testing faces.
2. Apply the configured sensor mapping and construct feature maps.
3. Fit a two-component GMM for each feature channel to estimate point-wise
   damage probability.
4. Fuse GMM probability outputs using Bayesian fusion.
5. Export probability maps, binary masks, and optional metrics when a
   ground-truth MAT file is supplied.

## Main Outputs

The pipeline writes all results under `--output-root`:

| Output path | Description |
|---|---|
| `feature_extr/` | Extracted feature tables and feature maps. |
| `psd/` | Frequency-domain feature tables and maps. |
| `features/` | Combined GMM input feature maps. |
| `GMM_results/` | GMM probability arrays, masks, and feature-order metadata. |
| `bayes_results_2/` | Bayesian fused maps and masks. |
| `tables/` | Minimal run summary and normalization coefficients. |

## Notes

- Input data are not included in this repository.
- Dataset-specific paths and optional settings should be supplied by the user at runtime.
- Large generated outputs should not be committed to the public repository.
