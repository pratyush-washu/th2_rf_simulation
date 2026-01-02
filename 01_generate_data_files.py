"""
01_generate_data_files.py - Generate synapse and stimulus grid data files

This script generates all necessary pickle files for running the simulation:
1. Synapse distribution data (positions along dendrites)
2. Stimulus grid data (grid positions and synapse masks)

These files are saved to the data/ folder and are required before running
the simulation script (02_run_simulation.py).

Usage:
    python 01_generate_data_files.py

Author: Pratyush Ramakrishna
Date: 2025
"""

from __future__ import division
import os
import sys
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuron import h
h.load_file('stdrun.hoc')
h.load_file('stdlib.hoc')
h.load_file('import3d.hoc')

from base_functions import distribute_synapses_uniform_perturbed, load_swc, grid_synapses_pruned

# =============================================================================
# CONFIGURATION - Modify these parameters as needed
# =============================================================================

# Cell ID and file paths
CELL_ID = 11
SWC_FILENAME = f'cell_{CELL_ID}_updated_soma.swc'

# Directory paths (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MORPHOLOGY_DIR = os.path.join(SCRIPT_DIR, 'morphology')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# Synapse distribution parameters
SYNAPSE_DENSITY = 0.064  # synapses per μm of dendrite

# Stimulus grid parameters
GRID_SIZE = 50  # μm between grid centers
REGION_SIZE = 100  # μm, size of stimulus region

# Stimulus pattern parameters (Gaussian-smoothed square)
FRAME_SIZE = 100  # pixels
SQUARE_SIZE = 50  # pixels (size of bright square)
GAUSSIAN_SIGMA = 25  # pixels (blur amount)

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("=" * 60)
    print("TH2 Receptive Field Simulation - Data File Generator")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load mechanism DLL if available
    mechanism_dll = os.path.join(SCRIPT_DIR, 'mechanisms', 'nrnmech.dll')
    if os.path.exists(mechanism_dll):
        try:
            h.nrn_load_dll(mechanism_dll)
            print(f"Loaded mechanism DLL: {mechanism_dll}")
        except Exception as e:
            print(f"Warning: Could not load mechanism DLL: {e}")
    
    # Path to SWC file
    swc_path = os.path.join(MORPHOLOGY_DIR, SWC_FILENAME)
    
    if not os.path.exists(swc_path):
        print(f"\nERROR: SWC file not found: {swc_path}")
        print("  Please copy your SWC file to the morphology/ folder")
        sys.exit(1)
    
    print(f"\n[1/4] Loading cell morphology from: {SWC_FILENAME}")
    cell, secs_list = load_swc(swc_path)
    print(f"      Loaded {len(cell.all)} sections")
    
    # Distribute synapses
    print(f"\n[2/4] Distributing synapses (density = {SYNAPSE_DENSITY} syn/μm)")
    secs_off, synapses_off, syn_x_off, syn_y_off = distribute_synapses_uniform_perturbed(
        cell, density=SYNAPSE_DENSITY
    )
    
    total_synapses = sum(len(s) for s in synapses_off)
    print(f"      Created {total_synapses} synapses")
    
    # Save synapse data
    synapse_data = {
        'synapses_off': synapses_off,
        'syn_x_off': syn_x_off,
        'syn_y_off': syn_y_off
    }
    
    synapse_file = os.path.join(DATA_DIR, f'synapse_data_cell_{CELL_ID}.pkl')
    with open(synapse_file, 'wb') as f:
        pickle.dump(synapse_data, f)
    print(f"      Saved to: synapse_data_cell_{CELL_ID}.pkl")
    
    # Create stimulus grid
    print(f"\n[3/4] Creating stimulus grid (grid_size = {GRID_SIZE} μm)")
    x_po, y_po, po_list, syn_masks_stim = grid_synapses_pruned(
        syn_x_off, syn_y_off, secs_off, 
        grid_size=GRID_SIZE, 
        region_size=REGION_SIZE
    )
    
    print(f"      Grid positions: {len(x_po)}")
    print(f"      Positions with synapses: {len(po_list)}")
    
    # Create stimulus pattern (Gaussian-smoothed square)
    print(f"\n[4/4] Creating stimulus pattern (Gaussian sigma = {GAUSSIAN_SIGMA})")
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE))
    start = (FRAME_SIZE - SQUARE_SIZE) // 2
    end = start + SQUARE_SIZE
    frame[start:end, start:end] = 1
    
    convolved_frame = gaussian_filter(frame, sigma=GAUSSIAN_SIGMA)
    convolved_frame = (convolved_frame - np.min(convolved_frame)) / \
                      (np.max(convolved_frame) - np.min(convolved_frame))
    convolved_frame = np.round(convolved_frame, 3)
    
    # Save stimulus grid data
    stim_data = (x_po, y_po, po_list, syn_masks_stim, convolved_frame)
    stim_file = os.path.join(DATA_DIR, f'stimulus_grid_cell_{CELL_ID}.pkl')
    with open(stim_file, 'wb') as f:
        pickle.dump(stim_data, f)
    print(f"      Saved to: stimulus_grid_cell_{CELL_ID}.pkl")
    
    # Save position list for reference
    pos_file = os.path.join(DATA_DIR, f'position_list_cell_{CELL_ID}.txt')
    np.savetxt(pos_file, po_list, fmt='%d')
    print(f"      Position list saved to: position_list_cell_{CELL_ID}.txt")
    
    # Summary
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nGenerated files in {DATA_DIR}:")
    print(f"  - synapse_data_cell_{CELL_ID}.pkl")
    print(f"  - stimulus_grid_cell_{CELL_ID}.pkl")
    print(f"  - position_list_cell_{CELL_ID}.txt")
    print(f"\nNumber of stimulus positions to simulate: {len(po_list)}")
    print("\nNext step: Run 02_run_simulation.py to simulate responses")


if __name__ == '__main__':
    main()
