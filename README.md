<div align="center">

# ⚛️ MACE-ActiveLearning-SAC

**Autonomous active learning of machine-learned interatomic potentials for single-atom catalysis**

H₂ dissociation on nitrogen-doped graphene-supported palladium (Pd–C<sub>3−x</sub>N<sub>x</sub>)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![MACE](https://img.shields.io/badge/MLIP-MACE-6f42c1)
![ASE](https://img.shields.io/badge/ASE-enabled-0366d6)
![Jobflow](https://img.shields.io/badge/orchestration-jobflow-28a745)
![MongoDB](https://img.shields.io/badge/storage-MongoDB%20Atlas-47A248?logo=mongodb&logoColor=white)
[![Paper](https://img.shields.io/badge/paper-J.%20Phys.%20Chem.%20Lett.%202026-orange)](https://doi.org/10.1021/acs.jpclett.5c03805)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Overview](#-overview) · [Pipeline](#-pipeline-architecture) · [Results](#-key-results) · [Quick start](#-quick-start) · [Structure](#-repository-structure) · [Publication](#-related-publication--citation) · [License](#-license)

</div>

---

## 📖 Overview

This repository hosts an autonomous, data-driven pipeline for evaluating the **kinetic and thermodynamic properties of single-atom catalysts (SACs)**. It investigates hydrogen (H₂) dissociation pathways on nitrogen-doped graphene-supported palladium.

The workflow runs an **active learning loop** that:

- 🧠 trains a Machine Learning Interatomic Potential (**MACE**) on DFT reference data,
- 🚏 automates **CI-NEB** (climbing-image Nudged Elastic Band) calculations,
- 📈 extracts **Brønsted–Evans–Polanyi (BEP)** scaling relations,

all without manual intervention.

> [!NOTE]
> **Scope of this repository.** This is a lightweight, laptop-scale demonstration of the *automated workflow and its mechanism* (active-learning loop, uncertainty-driven decisions, CI-NEB automation, BEP analysis). Calculation settings are deliberately inexpensive, so the numerical values shown below illustrate the method and are not converged production results. All settings are adjustable in [`config.yaml`](config.yaml) for higher-precision runs.

> 📄 **This work is referenced to the following publication:**
> S. Ziat, F. Brix, A. Tsaturyan, B. Kierren, É. Gaudry, *"How N-Doping Promotes Hydrogen Dissociation at Graphene-Based Single-Atom Catalysts"*, **J. Phys. Chem. Lett.**, 2026. [doi:10.1021/acs.jpclett.5c03805](https://doi.org/10.1021/acs.jpclett.5c03805)

---

## 🔧 Pipeline Architecture

The pipeline uses [`jobflow`](https://github.com/materialsproject/jobflow) to orchestrate tasks and make dynamic decisions based on model confidence.

### Active learning loop

```mermaid
graph TD
    classDef start_end fill:#28a745,stroke:#fff,stroke-width:2px,color:#fff;
    classDef decision fill:#ffc107,stroke:#fff,stroke-width:2px,color:#000;
    classDef process fill:#0366d6,stroke:#fff,stroke-width:2px,color:#fff;
    classDef analysis fill:#6f42c1,stroke:#fff,stroke-width:2px,color:#fff;

    Start((Start Workflow)):::start_end --> Init[Load State: Smart Resume]:::process
    Init --> LoopStart{Active Learning Loop}:::decision

    LoopStart --> CINEB[run_mlip_cineb_job<br>Execute NEB with current MACE]:::process
    CINEB --> Eval[active_learning_decision_job<br>Evaluate uncertainty]:::process

    Eval --> Check{Are all barriers<br>confident?}:::decision

    Check -->|No| DFT[run_dft_reference_job<br>Run AIMD on uncertain images]:::process
    DFT --> Train[train_mace_mlip_job<br>Retrain MACE on new data]:::process
    Train --> Validate[evaluate_mace_model_job<br>Calculate MAE on test set]:::process
    Validate --> LoopStart

    Check -->|Yes| BEP[analyze_bep_scaling_job<br>Extract Ea & dE]:::analysis
    BEP --> Plot[Generate BEP & Parity Figures]:::analysis
    Plot --> End((Workflow Complete)):::start_end
```

### Data synchronization flow

```mermaid
graph LR
    classDef db fill:#dbab0a,stroke:#fff,stroke-width:2px,color:#000;
    classDef script fill:#24292e,stroke:#fff,stroke-width:2px,color:#fff;
    classDef output fill:#ea4a5a,stroke:#fff,stroke-width:2px,color:#fff;

    subgraph Local["Local Execution Environment"]
        WF[run_workflow.py]:::script -->|Reads/Writes| LDB[(catalysis.db)]:::db
        WF -->|Generates| TRAJ[NEB Trajectories & Logs]:::output
        WF -->|Generates| FIGS[Figures & Checkpoints]:::output

        SDB[sync_catalysis_db.py]:::script -.->|Reads| LDB
        SA[sync_atlas.py]:::script -.->|Reads| LDB
    end

    subgraph Cloud["Cloud Infrastructure"]
        SA ==>|Pushes Collections| MDB[(MongoDB Atlas)]:::db
    end
```

---

## 📊 Key Results

As a demonstration, the automated workflow extracted the kinetic barriers for heterolytic and homolytic H₂ dissociation across several Pd coordination motifs. In this run, nitrogen-rich motifs show lower barriers than the all-carbon Pd–C₃ site.

| Motif | Activation barrier *E<sub>a</sub>* (eV) | Reaction energy *ΔE* (eV) |
| :--- | :---: | :---: |
| **Pd–N₂C₁** | **0.474** | 0.424 |
| Pd–N₃ | 0.603 | 0.529 |
| Pd–N₁C₂ | 0.982 | 0.609 |
| Pd–C₃ | 1.134 | 1.005 |

### BEP scaling relation

The pipeline automatically verified a linear Brønsted–Evans–Polanyi relationship across the dataset:

$$E_a = 1.08\,\Delta E + 0.10 \qquad (R^2 = 0.779)$$

---

## 🖼️ Pipeline Visualizations

Figures are generated into [`figures/`](figures/). Full-resolution vector versions of the parity plots: [adsorption](figures/mace_adsorption_parity-1.png) · [total energy](figures/mace_total_energy_parity-1.png).

### 1. BEP scaling

Linear scaling between thermodynamic reaction energy and kinetic activation barrier across the sampled motifs.

<p align="center">
  <img src="figures/bep_scaling_validation.png" alt="BEP scaling relation" width="600">
</p>

### 2. Kinetic profiles (CI-NEB)

Reaction pathways for heterolytic and homolytic H₂ dissociation on the active sites, generated by MACE-driven CI-NEB.

<p align="center">
  <img src="figures/neb_energy_profiles.png" alt="CI-NEB energy profiles" width="600">
</p>

### 3. MACE model accuracy

Parity plots of MACE against DFT reference data.

<table align="center">
  <tr>
    <td align="center"><b>Adsorption energy parity</b><br><img src="figures/mace_adsorption_parity-1.png" width="380"></td>
    <td align="center"><b>Total energy parity</b><br><img src="figures/mace_total_energy_parity-1.png" width="380"></td>
  </tr>
</table>

### 4. Training convergence

Loss, energy/force RMSE and parity plots from a MACE training run (checkpoint loaded from epoch 59). Training was kept short on purpose to demonstrate the loop on a laptop; longer training and larger datasets improve accuracy.

<p align="center">
  <img src="results/mace_multitm_model_run-123_train_Default_stage_one.png" alt="MACE training convergence: loss, RMSE and parity plots" width="800">
</p>

---

## 🚀 Quick Start

### Installation

Requires Python 3.10+ and a configured Conda environment. The pipeline relies on [ASE](https://wiki.fysik.dtu.dk/ase/), [MACE](https://github.com/ACEsuit/mace) and [Jobflow](https://github.com/materialsproject/jobflow).

```bash
git clone https://github.com/safouanziat/MACE-ActiveLearning-SAC.git
cd MACE-ActiveLearning-SAC

# Install core dependencies
pip install ase jobflow mace-torch matplotlib numpy
```

### Usage

**1. Run the active learning pipeline**

Jobflow's *Smart Resume* is enabled: interrupted runs automatically skip previously completed DFT and MACE training steps.

```bash
python run_workflow.py 2>&1 | tee execution_pipeline.log
```

**2. Synchronize databases**

```bash
python sync_catalysis_db.py
python sync_atlas.py
```

> [!CAUTION]
> Do **not** run the synchronization scripts concurrently with `run_workflow.py`. Doing so causes `database is locked` conflicts. Run them only after the pipeline has finished or been safely paused.

### Configuration

Hyperparameters and tolerances (e.g. uncertainty thresholds for the active learning decision) live in [`config.yaml`](config.yaml).

---

## 📁 Repository Structure

```text
.
├── checkpoints/              # Saved model weights during MACE training
├── figures/                  # Generated plots (parity, BEP scaling, energy profiles)
├── gpaw_logs/                # Outputs from DFT reference calculations
├── results/                  # MACE training diagnostics (loss, RMSE, parity)
├── structures_neb/           # Initial and final state .xyz/.traj geometries
├── project2_workflow.py      # Jobflow orchestration and node definitions
├── run_workflow.py           # Main execution entry point for the active learning loop
├── sync_atlas.py             # MongoDB Atlas synchronization script
├── sync_catalysis_db.py      # Local database management script
└── config.yaml               # Pipeline hyperparameters and tolerances
```

---

## 📚 Related Publication & Citation

This repository is associated with the study of H₂ dissociation on N-doped graphene-supported single-atom catalysts reported in:

> **S. Ziat**, F. Brix, A. Tsaturyan, B. Kierren, É. Gaudry,
> *How N-Doping Promotes Hydrogen Dissociation at Graphene-Based Single-Atom Catalysts*,
> **J. Phys. Chem. Lett.**, 2026. [doi:10.1021/acs.jpclett.5c03805](https://doi.org/10.1021/acs.jpclett.5c03805)

If you use this pipeline or its results, please cite the paper:

```bibtex
@article{ziat2026ndoping,
  author  = {Ziat, Safouan and Brix, F. and Tsaturyan, A. and Kierren, B. and Gaudry, {\'E}.},
  title   = {How N-Doping Promotes Hydrogen Dissociation at Graphene-Based Single-Atom Catalysts},
  journal = {The Journal of Physical Chemistry Letters},
  year    = {2026},
  doi     = {10.1021/acs.jpclett.5c03805}
}
```

and, if relevant, the repository itself:

```bibtex
@software{ziat2026mace_activelearning_sac,
  author = {Ziat, Safouan},
  title  = {MACE-ActiveLearning-SAC: Autonomous Active Learning of MLIPs for Single-Atom Catalysis},
  url    = {https://github.com/safouanziat/MACE-ActiveLearning-SAC},
  year   = {2026}
}
```

---

## 📄 License

Released under the [MIT License](LICENSE). Copyright © 2026 Safouan Ziat.

---

## 👤 Author

**Safouan Ziat** · [GitHub](https://github.com/safouanziat)
