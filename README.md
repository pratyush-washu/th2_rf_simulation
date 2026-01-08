# Biophysical Model of TH2 Amacrine Cell Receptive Fields

This repository contains a multi-compartmental model of TH2 amacrine cells in the mouse retina, implemented in Python/NEURON. The code simulates dendritic integration and maps receptive fields (RF) to investigate the "bipartite" physiology of TH2 cells (local proximal vs. global distal integration).

## Overview

The model uses morphological reconstructions from 3DEM data and biophysical constraints derived from electrophysiology to simulate:

* **Morphology:** Realistic SWC reconstruction with diameter constraints derived from ultrastructural 3DEM measurements.
* **Biophysics:** Active dendritic integration using voltage-gated sodium channels ($Na_V$) and passive leak channels.
* **Synaptic Input:** Uniform distribution of excitatory synapses (0.064 syn/µm) mimicking OFF bipolar cell innervation.
* **Receptive Field Mapping:** Grid-based visual stimulation to calculate RF size and center-of-mass (COM) offsets across the dendritic arbor.

## Simulation Pipeline

The workflow consists of three sequential steps:

### 1. Data Generation
* **Script:** `01_generate_data_files.py`
* **Function:** Loads SWC morphology, distributes synapses uniformly (with noise perturbation) along dendrites, and generates the stimulus grid (50x50 µm patches).
* **Output:** Serialized synapse and grid data in `data/`.

### 2. Simulation
* **Script:** `02_run_simulation.py`
* **Function:** Runs NEURON simulations for every grid position. Solves the cable equation to compute membrane potential, incorporating bi-exponential synaptic inputs (Exp2Syn) and voltage-gated sodium currents ($I_{Na}$) modeled via Hodgkin-Huxley formalism.
* **Output:** Maximum voltage depolarization at all dendritic segments in `output/`.

### 3. Analysis
* **Script:** `03_analyze_receptive_field.py`
* **Function:** Reconstructs RFs from voltage responses. Calculates RF size (major axis at 25% threshold) and spatial offset (distance from recording site to RF COM). Compares proximal (<80 µm) vs. distal dendritic integration.
* **Output:** Hexbin plots and summary statistics in `figures/`.

## Model Parameters

Parameters were tuned via grid search to match physiological length constants ($\lambda$) and voltage responses observed *in vivo*.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **$C_m$** | 1.0 $\mu F/cm^2$ | Membrane capacitance |
| **$R_a$** | 60 \Omega \text{cm} | Axial Resistance |
| **$g_{leak}$** | 1.2 $mS/cm^2$ | Passive leak conductance |
| **$g_{Na}$** | 1.9 $mS/cm^2$ | Voltage-gated $Na^+$ conductance density |
| **$E_{leak}$** | -55 mV | Leak reversal potential |
| **$E_{Na}$** | 50 mV | Sodium reversal potential |
| **$\lambda$** | 460 µm | Effective electrotonic length constant |
| **Synapse Density** | 0.064 syn/µm | Excitatory OFF bipolar input density |

## Requirements

* Python 3.x
* NEURON (compiled with Python interface)
* NumPy, SciPy
* scikit-image
* Matplotlib
* Custom $Na_V$ channel mechanism in NEURON (compile `NaV_pr.mod` file in `mechanisms/`)