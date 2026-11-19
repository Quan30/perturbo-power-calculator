# PerTurbo Power Calculator

An interactive app for planning and interpreting **Perturb-seq / single-cell CRISPR screens**.
It estimates the statistical power to detect a change in gene expression after a perturbation,
and how that power changes with **cells per guide**, **sequencing depth**, **effect size** and
**baseline expression**.

It answers three questions:

| | Question | Example output (default settings, bundled data) |
|---|---|---|
| **Analysis 1: Cis-regulatory power** | *Which genes near my enhancer or variant could this screen detect if the enhancer changed their expression?* | "5 of 19 evaluable genes within 1,000 kb are expected to reach ≥ 80 % power for an assumed 50 % decrease." |
| **Analysis 2: Target-program (regulon) power** | *If I perturb this transcription factor, what fraction of its target genes could I detect?* | "1,471 of 4,447 evaluable GATA1 targets are expected to reach ≥ 80 % power for a 50 % decrease." |
| **Analysis 3: Screen-wide power** | *Across a whole screen, what fraction of perturbation–gene pairs is adequately powered, and how much do more cells or reads help?* | "1,046 of 7,056 eligible pairs (14.8 %) reach ≥ 80 % power at the selected design." |

The app ships with fitted reference baselines for two public K562 screens, so **all three
analyses work immediately after installation**. No GPU and no raw data are required.

* **Enhancer screen, high MOI**: Gasperini *et al.*, *Cell* 2019 (GEO GSE120861)
* **Genome-wide screen, low MOI**: Replogle *et al.*, *Cell* 2022

---

## Quick start (about 5 minutes)

Requirements: Python ≥ 3.10 and `git`. Works on Linux and macOS (Windows via WSL).

```bash
git clone https://github.com/Quan30/perturbo-power-calculator.git
cd perturbo-power-calculator

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash run_app.sh
```

Then open **http://localhost:8501** in your browser.

<details>
<summary><b>Running on a remote server / HPC node</b></summary>

Start the app on the server:

```bash
bash run_app.sh --server.address 0.0.0.0 --server.port 8501
```

On your laptop, forward the port, then open http://localhost:8501:

```bash
ssh -N -L 8501:localhost:8501 user@your-server
```
</details>

<details>
<summary><b>Using conda instead of venv</b></summary>

```bash
conda create -n perturbo-power python=3.12 -y
conda activate perturbo-power
pip install -r requirements.txt
bash run_app.sh
```
</details>

---

## Try each analysis

The home page has one card per analysis. Design settings (effect size, target power, cells per
guide, depth) are in the **sidebar**. Hover over the ⓘ icon next to any control for an
explanation.

### Analysis 1: Cis-regulatory power (enhancer/variant → nearby genes)

Uses the Gasperini enhancer screen. Nothing needs to be selected first.

1. Click **Open Analysis 1**.
2. **Dataset overview**: the fraction of all candidate enhancer–gene pairs reaching the target
   power, and how it changes with cells per guide, depth and expression.
3. **Gene-level analysis**: pick a gene to see all enhancers tested against it.
4. **Pair / locus-level analysis**: choose an enhancer, or switch **Locus type** to
   *Variant (SNP position)* and enter a chromosome and position. You get every gene in the window
   with its baseline expression, detection rate and power, plus power curves for a selected gene.

Defaults: 50 % assumed decrease, 80 % target power, 200 cells per guide, 1 Mb window.

> A variant is used **only as a genomic coordinate**. The app does not claim that it has a
> measured or causal effect.

### Analysis 2: Target-program power (TF → regulon / GRN)

1. Click **Open Analysis 2**. The Replogle baseline is selected by default. Use
   **Change baseline file** to pick another baseline, for example one you fitted yourself.
2. Choose the **target program source**:
   * **K562 built-in TF program**: TF targets from ReMap 2022 ChIP-seq peaks near promoters of
     genes expressed in K562 (Replogle baseline only). You can require more supporting peaks.
   * **CollecTRI curated regulon**: 1,201 regulators from CollecTRI (Müller-Dott *et al.* 2023),
     bundled for offline use.
   * **Custom gene list**: paste or upload gene symbols or Ensembl IDs.
3. The page reports how many targets reach the target power, how this changes with cells and
   depth, and which genes were unmatched or filtered out for low expression (default minimum
   0.1 counts per cell).

### Analysis 3: Screen-wide empirical power

1. Click **Open Analysis 3**. The bundled simulation results load automatically.
2. Global results come first: the fraction of pairs adequately powered and how it changes with
   cells per guide and depth (one curve per effect size). Next come the distribution of per-pair
   power and power by baseline-expression decile.
3. To inspect one pair, pick an enhancer and a gene, then click **Load selected pair**.

The bundled results are repeated end-to-end PerTurbo simulations of the Gasperini screen in cis
mode: 7,056 enhancer–gene pairs × 3 effect sizes (10/25/50 %) × 5 cell counts (100–2,000 cells
per guide) × 3 depths (0.5×/1×/2×), with 30 replicates per design. One design has 29 replicates
and is labelled provisional. The page reads these precomputed results and does not re-run
simulations.

---

## How power is calculated

* **Analyses 1 and 2** are analytical and instantaneous. Each gene's expression is modelled as
  negative-binomial, with gene-specific baseline mean `exp(β₀)` and dispersion `θ` taken from a
  PerTurbo fit to non-targeting control cells. Power comes from a Wald test of the assumed
  log fold-change at the chosen number of cells and depth.
* **Calibration.** Raw analytical power is optimistic compared with the full PerTurbo pipeline
  (empirical-null p-values + FDR). The app therefore applies effective z-thresholds calibrated
  against the repeated simulations (`artifacts/*.json`). It applies them only where they match
  the context: Gasperini cis decrease, and provisionally Gasperini cis increase and Replogle
  trans decrease. Every result page states which calibration, if any, was applied.
* **Analysis 3** reports **empirical** power: the fraction of simulated screens in which the pair
  was detected at FDR ≤ 10 %.
* **Depth** is entered as *expected modelled counts (UMIs) per cell*, not sequencing reads. The
  reference depth of each baseline is `Σ exp(β₀)`: about 16,600 for Gasperini and 10,300 for
  Replogle.

---

## Use your own data

| What you have | What works |
|---|---|
| A fitted baseline `.npz` (arrays `beta_0`, `theta`, `gene_names`; gene names stored as strings, not pickled objects) | Select it in **Data Setup** or under **Change baseline file** in Analysis 2. Analysis 2 then works with a custom or CollecTRI gene list. |
| Raw counts (`.h5ad` / `.h5mu`) | **Data Setup → Compute from raw data**: choose the file, map its count matrix, perturbation column and negative controls, then validate. Fitting the baseline **requires the PerTurbo model package** (see below). |
| Your own Analysis 3 simulation table | Set `CORTADO_ANALYSIS3_PAIR_POWER=/path/to/table.parquet` before launching. |

Analysis 1 needs element–gene coordinates and per-gene detection rates. These are currently
bundled for the Gasperini screen only.

### Fit a baseline from your own data

Fitting uses the PerTurbo model (JAX/NumPyro; a GPU is recommended). The model code is **not
part of this repository yet**; it will be released with the manuscript. Without it, the fitting
step shows a clear "model package not installed" message. Every other part of the app works
without it. If you have access to the model package:

```bash
pip install /path/to/model-package          # provides `import cortado`
pip install --upgrade "jax[cuda12]"          # optional: NVIDIA GPU support
python -c "import cortado, jax; print(jax.devices())"
```

Fits can also be run from the command line:

```bash
python scripts/fit_baseline.py --help
```

---

## Repository layout

```
app.py                  Home page (entry point)
run_app.sh              Launcher: sets default data paths, then starts Streamlit
pages/                  Data Setup and the three analysis pages
utils_analysis/         Power calculations (NB Wald power, Analyses 1–3, CollecTRI)
services/               Baseline discovery, background fit jobs, uploads, calibration
scripts/                fit_baseline.py, validate_power_calibration.py, check_repo_resources.py
baselines/              Fitted Gasperini and Replogle reference baselines
artifacts/              Gasperini pair table, gene detection rates, Analysis 3 results, calibrations
data/                   K562 TF programs and CollecTRI regulons
tests/                  Automated tests (pytest)
```

[`data/README.md`](data/README.md) documents the source of each bundled file.

### Advanced configuration

`run_app.sh` points the app at the bundled files. To use other files, set any of these
environment variables before launching. `CORTADO` is the internal name of the PerTurbo JAX
implementation.

| Variable | Purpose |
|---|---|
| `CORTADO_DATA_ROOT` | Folder scanned for baselines; fit jobs are written here (default `baselines/`) |
| `CORTADO_GASPERINI_BASELINE`, `CORTADO_REPLOGLE_BASELINE` | Default baselines for the two built-in screens |
| `CORTADO_GASPERINI_PAIR_TABLE` | Element–gene pair table for Analysis 1 |
| `CORTADO_GASPERINI_DETECTION_SUMMARY` | Per-gene detection table for Analysis 1 |
| `CORTADO_GASPERINI_GENOME_BUILD` | Genome build of the pair-table coordinates (e.g. `GRCh38`); needed for variant input |
| `CORTADO_K562_M2KB_PROGRAMS` | K562 TF program table for Analysis 2 |
| `CORTADO_ANALYSIS3_PAIR_POWER` | Replacement Analysis 3 simulation table |
| `CORTADO_GASPERINI_EFFECTS` | Optional observed effect-size table shown as context in Analysis 3 |

---

## Run the tests

```bash
pip install pytest
python -m pytest -q
```

---

## Current limitations

* **Analysis 3** results are bundled for the Gasperini screen in **cis** mode only. Trans-mode and
  Replogle screen-wide simulations are not yet included.
* The calibrations for Gasperini cis **increase** and Replogle trans **decrease** are provisional
  (10 replicates per design). Custom baselines are not calibrated; their results show raw
  analytical power.
* Genes are modelled independently. The calculator does not model gene–gene correlation or
  guide-to-guide differences in knockdown efficiency.
* CollecTRI regulons are used without sign or evidence filters. The assumed effect direction is
  the user's choice.

## Citation

If you use this calculator, please cite the PerTurbo manuscript (in preparation):

> Blaine L.\*, Lu J.\*, Jankowiak M., Ryu J., Ferreira da Silva L., Heinig M., Pinello L.
> *Robust differential expression and power analysis for genome-scale Perturb-seq with PerTurbo.*

## License

[MIT](LICENSE)
