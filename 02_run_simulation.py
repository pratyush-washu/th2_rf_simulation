"""
02_run_simulation.py - Run NEURON simulations for receptive field mapping

This script runs NEURON simulations at each stimulus grid position to
generate voltage response data for receptive field analysis.

Prerequisites:
    - Run 01_generate_data_files.py first to generate required data files
    - SWC morphology file in morphology/ folder
    - NaV mechanism DLL in mechanisms/ folder

Usage:
    python 02_run_simulation.py

Output:
    - Voltage response files for each grid position in output/ folder

Author: [Your Name]
Date: 2025
"""

from __future__ import division
import os
import sys
import pickle
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuron import h
h.load_file('stdrun.hoc')
h.load_file('stdlib.hoc')
h.load_file('import3d.hoc')

from base_functions import run_simulation_square, max_voltage_positions

# =============================================================================
# CONFIGURATION - Simulation parameters
# =============================================================================

# Cell ID
CELL_ID = 11
SWC_FILENAME = f'cell_{CELL_ID}_updated_soma.swc'

# Model parameters (from your optimization)
G_PAS = 0.36        # Passive leak conductance (S/cm²)
NAV_DENSITY = 0.16  # Sodium channel density (S/cm²)
WEIGHT = 0.27       # Synaptic weight scaling factor

# Output configuration
SIMULATION_INDEX = 1  # Index for output file naming

# Directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MORPHOLOGY_DIR = os.path.join(SCRIPT_DIR, 'morphology')
MECHANISMS_DIR = os.path.join(SCRIPT_DIR, 'mechanisms')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

# Analysis parameters
DISTANCE_THRESHOLD = 2000  # μm, max distance from stimulus center
DECIMALS = 3  # decimal precision for output

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("=" * 60)
    print("RGC Receptive Field Simulation - NEURON Simulator")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Enable variable timestep solver for efficiency
    cvode = h.CVode()
    cvode.active(1)
    
    # Load mechanism DLL
    mechanism_dll = os.path.join(MECHANISMS_DIR, 'nrnmech.dll')
    if os.path.exists(mechanism_dll):
        try:
            h.nrn_load_dll(mechanism_dll)
            print(f"Loaded NaV mechanism from: {mechanism_dll}")
        except Exception as e:
            print(f"ERROR: Could not load mechanism DLL: {e}")
            print("  Ensure nrnmech.dll is in the mechanisms/ folder")
            sys.exit(1)
    else:
        print(f"ERROR: Mechanism DLL not found: {mechanism_dll}")
        print("  Please copy nrnmech.dll to the mechanisms/ folder")
        sys.exit(1)
    
    # Check for SWC file
    swc_path = os.path.join(MORPHOLOGY_DIR, SWC_FILENAME)
    if not os.path.exists(swc_path):
        print(f"ERROR: SWC file not found: {swc_path}")
        sys.exit(1)
    print(f"Found morphology file: {SWC_FILENAME}")
    
    # Load synapse data
    synapse_file = os.path.join(DATA_DIR, f'synapse_data_cell_{CELL_ID}.pkl')
    if not os.path.exists(synapse_file):
        print(f"ERROR: Synapse data not found: {synapse_file}")
        print("  Run 01_generate_data_files.py first")
        sys.exit(1)
    
    with open(synapse_file, 'rb') as f:
        synapse_data = pickle.load(f)
    
    synapses_off = synapse_data['synapses_off']
    syn_x_off = synapse_data['syn_x_off']
    syn_y_off = synapse_data['syn_y_off']
    print(f"Loaded synapse data")
    
    # Load stimulus grid data
    stim_file = os.path.join(DATA_DIR, f'stimulus_grid_cell_{CELL_ID}.pkl')
    if not os.path.exists(stim_file):
        print(f"ERROR: Stimulus grid data not found: {stim_file}")
        print("  Run 01_generate_data_files.py first")
        sys.exit(1)
    
    with open(stim_file, 'rb') as f:
        stim_data = pickle.load(f)
    
    x_po, y_po, po_list, syn_masks_stim, convolved_frame = stim_data
    print(f"Loaded stimulus grid data")
    
    # Print simulation parameters
    print("\n" + "-" * 60)
    print("SIMULATION PARAMETERS")
    print("-" * 60)
    print(f"  Cell ID:          {CELL_ID}")
    print(f"  g_pas:            {G_PAS} S/cm²")
    print(f"  Nav density:      {NAV_DENSITY} S/cm²")
    print(f"  Synaptic weight:  {WEIGHT}")
    print(f"  Grid positions:   {len(po_list)}")
    print("-" * 60)
    
    # Run simulations
    print(f"\nRunning simulations for {len(po_list)} grid positions...")
    print("This may take several minutes.\n")
    
    completed = 0
    skipped = 0
    
    for i, ppp in enumerate(po_list):
        output_file = os.path.join(
            OUTPUT_DIR, 
            f'voltages_ind_{SIMULATION_INDEX}_pos_{ppp}.txt'
        )
        
        # Skip if already exists
        if os.path.exists(output_file):
            skipped += 1
            continue
        
        # Progress indicator
        print(f"  Position {i+1}/{len(po_list)} (grid index {ppp})...", end='', flush=True)
        
        try:
            # Run simulation
            volt_list = run_simulation_square(
                swc_file=swc_path,
                g_pas=G_PAS,
                ppp=ppp,
                we=WEIGHT,
                syn_masks_stim_off=syn_masks_stim,
                syn_x_off=syn_x_off,
                syn_y_off=syn_y_off,
                synapses_off=synapses_off,
                nav_density=NAV_DENSITY,
                x_pos=x_po,
                y_pos=y_po,
                convolved_frame=convolved_frame
            )
            
            # Extract max voltages at each position
            x_coords, y_coords, volt_vals = max_voltage_positions(
                volt_list, 
                x_po[ppp], 
                y_po[ppp], 
                distance_threshold=DISTANCE_THRESHOLD, 
                decimals=DECIMALS
            )
            
            # Save results
            np.savetxt(
                output_file,
                np.column_stack((x_coords, y_coords, volt_vals)),
                fmt='%.2f,%.2f,%.2f',
                delimiter=','
            )
            
            completed += 1
            print(" done")
            
        except Exception as e:
            print(f" ERROR: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"\n  Completed:  {completed} simulations")
    print(f"  Skipped:    {skipped} (already exist)")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("\nNext step: Run 03_analyze_receptive_field.py to analyze results")


if __name__ == '__main__':
    main()
