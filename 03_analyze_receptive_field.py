"""
03_analyze_receptive_field.py - Analyze and plot receptive field properties

This script analyzes the simulation output to compute:
- Receptive field center of mass (COM) for each dendritic location
- Receptive field size for each dendritic location
- Proximal vs distal comparisons

Output:
- Hexbin plots showing RF size and COM across the dendritic tree
- Summary statistics saved to output/ folder

Prerequisites:
    - Run 01_generate_data_files.py first
    - Run 02_run_simulation.py first

Usage:
    python 03_analyze_receptive_field.py

Author: Pratyush Ramakrishna
Date: 2025
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csr_matrix
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, square
from skimage.draw import line
from skimage.transform import resize
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

# =============================================================================
# CONFIGURATION
# =============================================================================

# Cell ID
CELL_ID = 11
SWC_FILENAME = f'cell_{CELL_ID}_updated_soma.swc'

# Model parameters (for documentation in output)
G_PAS = 0.36
NAV_DENSITY = 0.16
WEIGHT = 0.27
SIMULATION_INDEX = 1

# Analysis parameters
PROXIMAL_THRESHOLD = 80  # μm from soma for proximal/distal classification
RF_THRESHOLD_LOW = 25    # % of max for RF boundary
RF_THRESHOLD_HIGH = 98   # % of max for RF peak
BRIDGE_THICKNESS = 3     # pixels for connecting RF components

# Directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MORPHOLOGY_DIR = os.path.join(SCRIPT_DIR, 'morphology')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')

# =============================================================================
# CUSTOM COLORMAP
# =============================================================================

# Green-White-Red colormap for visualization
color_green = (20/255, 181/255, 58/255)
color_white = (1, 1, 1)
color_red = (206/255, 17/255, 38/255)
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "custom_colormap", [color_green, color_white, color_red]
)

# =============================================================================
# FUNCTIONS
# =============================================================================

def plot_hexbin(image, vmin, vmax, title, ax=None):
    """
    Create a hexbin plot of spatial data.
    
    Parameters
    ----------
    image : ndarray
        2D array with values to plot (NaN for missing)
    vmin, vmax : float
        Color scale limits
    title : str
        Plot title
    ax : matplotlib axis, optional
        Axis to plot on
    """
    if ax is None:
        ax = plt.gca()
    
    x = np.tile(np.arange(image.shape[1]), image.shape[0])
    y = np.repeat(np.arange(image.shape[0]), image.shape[1])
    z = image.flatten()
    mask = ~np.isnan(z)
    
    hb = ax.hexbin(
        x[mask], y[mask], C=z[mask], 
        gridsize=20, cmap=CUSTOM_CMAP, 
        edgecolors='black', 
        reduce_C_function=np.nanmean, 
        vmin=vmin, vmax=vmax
    )
    
    return hb


def load_swc_coordinates(swc_path):
    """Load dendritic coordinates from SWC file."""
    swc_x = []
    swc_y = []
    x_y_vals = np.loadtxt(swc_path)
    
    for i in range(len(x_y_vals)):
        if x_y_vals[i][1] == 3:  # Type 3 = dendrite
            swc_x.append(x_y_vals[i][2])
            swc_y.append(x_y_vals[i][3])
    
    swc_x = np.round(np.array(swc_x), 0)
    swc_y = np.round(np.array(swc_y), 0)
    soma_pos = [swc_x[0], swc_y[0]]
    
    return swc_x, swc_y, soma_pos


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("=" * 60)
    print("RGC Receptive Field Analysis")
    print("=" * 60)
    
    # Ensure output directories exist
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load SWC coordinates
    swc_path = os.path.join(MORPHOLOGY_DIR, SWC_FILENAME)
    if not os.path.exists(swc_path):
        print(f"ERROR: SWC file not found: {swc_path}")
        sys.exit(1)
    
    print(f"\n[1/6] Loading morphology coordinates...")
    swc_x, swc_y, soma_pos = load_swc_coordinates(swc_path)
    print(f"      Loaded {len(swc_x)} dendritic points")
    print(f"      Soma position: ({soma_pos[0]:.0f}, {soma_pos[1]:.0f})")
    
    # Load stimulus grid data
    print(f"\n[2/6] Loading stimulus grid data...")
    stim_file = os.path.join(DATA_DIR, f'stimulus_grid_cell_{CELL_ID}.pkl')
    if not os.path.exists(stim_file):
        print(f"ERROR: Stimulus data not found. Run 01_generate_data_files.py first")
        sys.exit(1)
    
    with open(stim_file, 'rb') as f:
        stim_data = pickle.load(f)
    
    x_po, y_po, po_list, syn_masks_stim, convolved_frame = stim_data
    x_poos = np.unique(x_po)
    y_poos = np.unique(y_po)
    po_list = np.array(po_list)
    
    print(f"      Grid positions: {len(po_list)}")
    
    # Load and compile voltage data
    print(f"\n[3/6] Loading simulation data...")
    
    # Initialize data array
    data_arr = np.full((len(swc_x), len(po_list) + 2), np.nan)
    data_arr[:, 0] = swc_x
    data_arr[:, 1] = swc_y
    
    po_list_index = {ppp: idx for idx, ppp in enumerate(po_list)}
    files_loaded = 0
    
    for ppp in po_list:
        file_path = os.path.join(
            OUTPUT_DIR, 
            f'voltages_ind_{SIMULATION_INDEX}_pos_{ppp}.txt'
        )
        
        if not os.path.exists(file_path):
            continue
        
        volts = np.loadtxt(file_path, delimiter=',')
        ind_ppp = po_list_index[ppp]
        
        for row in volts:
            rowx = np.round(row[0], 0)
            rowy = np.round(row[1], 0)
            row_index = np.where((swc_x == rowx) & (swc_y == rowy))[0]
            if len(row_index) > 0:
                data_arr[row_index[0], ind_ppp + 2] = row[2]
        
        files_loaded += 1
    
    print(f"      Loaded {files_loaded} voltage files")
    
    if files_loaded == 0:
        print("ERROR: No voltage files found. Run 02_run_simulation.py first")
        sys.exit(1)
    
    # Remove rows with all NaN values
    data_arr = data_arr[~np.isnan(data_arr[:, 2:-1]).all(axis=1)]
    
    # Build response matrix
    print(f"\n[4/6] Building response matrix...")
    
    data_matrix = np.zeros((1400, 1400, len(po_list)))
    counts_matrix = np.zeros((1400, 1400, len(po_list)))
    
    for i in range(len(data_arr)):
        x, y = int(data_arr[i, 0]), int(data_arr[i, 1])
        data_matrix[y, x, :] += data_arr[i, 2:]
        counts_matrix[y, x, :] += 1
    
    # Average where we have multiple values
    data_matrix_2 = np.divide(data_matrix, counts_matrix, where=counts_matrix != 0)
    
    # Build ROI response matrix
    non_zero_count = np.count_nonzero(data_matrix_2[:, :, 0])
    resp_matrix = np.zeros((len(x_poos), len(y_poos), non_zero_count))
    roi_cent = []
    
    x_indices_dict = {xval: np.where(x_po == xval)[0] for xval in x_poos}
    y_indices_dict = {yval: np.where(y_po == yval)[0] for yval in y_poos}
    
    ccc = 0
    for oo in range(data_matrix_2.shape[0]):
        for pp in range(data_matrix_2.shape[1]):
            if np.any(data_matrix_2[oo, pp, :]):
                updated = False
                
                for i, xval in enumerate(x_poos):
                    for j, yval in enumerate(y_poos):
                        x_indices = x_indices_dict.get(xval, [])
                        y_indices = y_indices_dict.get(yval, [])
                        
                        if len(x_indices) > 0 and len(y_indices) > 0:
                            intersection = np.intersect1d(x_indices, y_indices)
                            
                            if len(intersection) > 0:
                                int_position = int(intersection[0])
                                
                                if int_position in po_list:
                                    idx_in_po_list = np.where(po_list == int_position)[0][0]
                                    
                                    if data_matrix_2[oo, pp, idx_in_po_list] != 0:
                                        if not updated:
                                            roi_cent.append([oo, pp])
                                            updated = True
                                        resp_matrix[i, j, ccc] = data_matrix_2[oo, pp, idx_in_po_list]
                
                if updated:
                    ccc += 1
    
    resp_matrix[resp_matrix == 0] = np.nan
    print(f"      ROI count: {len(roi_cent)}")
    
    # Normalize response matrix
    resp_dev_matrix = resp_matrix + 55  # Shift from resting potential
    max_v = np.nanmax(resp_dev_matrix[:])
    
    multple_max = np.nanmax(np.nanmax(resp_dev_matrix, axis=1), axis=0)
    repeated_max = np.tile(multple_max, (resp_dev_matrix.shape[0], resp_dev_matrix.shape[1], 1))
    
    resp_dev_matrix = 100 * (resp_dev_matrix / repeated_max)
    resp_dev_matrix[np.isnan(resp_dev_matrix)] = 0
    
    # Interpolate to higher resolution
    new_dim0 = resp_dev_matrix.shape[0] * 5
    new_dim1 = resp_dev_matrix.shape[1] * 5
    
    interpolated_matrix = resize(
        resp_dev_matrix,
        (new_dim0, new_dim1, resp_dev_matrix.shape[2]),
        order=1,
        mode='constant',
        anti_aliasing=True
    )
    interpolated_matrix[interpolated_matrix == 0] = np.nan
    
    # Analyze RF properties
    print(f"\n[5/6] Analyzing receptive field properties...")
    
    com_image = np.full((1400, 1400), np.nan)
    size_image = np.full((1400, 1400), np.nan)
    peak_image = np.full((1400, 1400), np.nan)
    
    for i in range(interpolated_matrix.shape[2]):
        immm = interpolated_matrix[:, :, i]
        
        # Binary masks
        binary_image = (immm > RF_THRESHOLD_LOW) & np.isfinite(immm)
        binary_image_2 = (immm > RF_THRESHOLD_HIGH) & np.isfinite(immm)
        
        # Label components
        labeled = label(binary_image)
        regions = regionprops(labeled)
        
        # Connect multiple components if needed
        if len(regions) > 1:
            cents = np.array([r.centroid for r in regions])
            D = cdist(cents, cents)
            mst = minimum_spanning_tree(D).toarray()
            
            merged_mask = binary_image.copy()
            bridge = np.zeros_like(merged_mask, dtype=bool)
            ys, xs = np.where(mst > 0)
            
            for a, b in zip(ys, xs):
                y1, x1 = cents[a]
                y2, x2 = cents[b]
                rr, cc = line(int(round(y1)), int(round(x1)), 
                             int(round(y2)), int(round(x2)))
                m = (rr >= 0) & (rr < merged_mask.shape[0]) & \
                    (cc >= 0) & (cc < merged_mask.shape[1])
                bridge[rr[m], cc[m]] = True
            
            if BRIDGE_THICKNESS > 1:
                bridge = binary_dilation(bridge, square(BRIDGE_THICKNESS))
            
            merged_mask |= bridge
            labeled = label(merged_mask)
            regions = regionprops(labeled)
        else:
            merged_mask = binary_image
        
        labeled_2 = label(binary_image_2)
        regions_2 = regionprops(labeled_2)
        
        if len(regions) > 0:
            region = max(regions, key=lambda r: r.area)
            major_axis_length = region.major_axis_length
            y0, x0 = region.centroid
            
            x_pos_roi, y_pos_roi = roi_cent[i]
            y0_remap = y0 * 10
            x0_remap = x0 * 10
            
            if len(regions_2) > 0:
                y0peak, x0peak = regions_2[0].centroid
                y0peak_remap = y0peak * 10
                x0peak_remap = x0peak * 10
            else:
                y0peak_remap, x0peak_remap = y0_remap, x0_remap
            
            com_image[y_pos_roi, x_pos_roi] = np.hypot(
                y_pos_roi - y0_remap, x_pos_roi - x0_remap
            )
            size_image[y_pos_roi, x_pos_roi] = major_axis_length * 10
            peak_image[y_pos_roi, x_pos_roi] = np.hypot(
                y_pos_roi - y0peak_remap, x_pos_roi - x0peak_remap
            )
    
    # Calculate proximal vs distal statistics
    prox_size, dist_size = [], []
    prox_com, dist_com = [], []
    
    for i in range(size_image.shape[0]):
        for j in range(size_image.shape[1]):
            pix_val = size_image[i, j]
            com_val = com_image[i, j]
            dist_from_soma = np.sqrt((i - soma_pos[0])**2 + (j - soma_pos[1])**2)
            
            if dist_from_soma < PROXIMAL_THRESHOLD:
                prox_size.append(pix_val)
                prox_com.append(com_val)
            else:
                dist_size.append(pix_val)
                dist_com.append(com_val)
    
    prox_size_median = np.round(np.nanmedian(prox_size), 2)
    dist_size_median = np.round(np.nanmedian(dist_size), 2)
    prox_com_median = np.round(np.nanmedian(prox_com), 2)
    dist_com_median = np.round(np.nanmedian(dist_com), 2)
    
    print(f"      Proximal RF size (median): {prox_size_median:.0f} μm")
    print(f"      Distal RF size (median):   {dist_size_median:.0f} μm")
    print(f"      Proximal COM (median):     {prox_com_median:.0f} μm")
    print(f"      Distal COM (median):       {dist_com_median:.0f} μm")
    print(f"      Max voltage deflection:    {max_v - 55:.1f} mV")
    
    # Generate plots
    print(f"\n[6/6] Generating figures...")
    
    # Plot 1: COM distance
    fig, ax = plt.subplots(figsize=(10, 8))
    hb = plot_hexbin(com_image.T, vmin=0, vmax=300, title='COM Distance', ax=ax)
    plt.colorbar(hb, ax=ax, label='COM Distance (μm)')
    plt.title(f'Receptive Field Center of Mass - Cell {CELL_ID}')
    plt.text(0.5, 0.01, 
             f'Proximal COM: {prox_com_median:.0f} μm, Distal COM: {dist_com_median:.0f} μm, Max ΔV: {max_v-55:.0f} mV',
             transform=plt.gca().transAxes, ha='center', fontsize=10)
    plt.axis('off')
    
    com_fig_path = os.path.join(FIGURES_DIR, f'com_distance_cell_{CELL_ID}.png')
    plt.savefig(com_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved: com_distance_cell_{CELL_ID}.png")
    
    # Plot 2: RF size
    fig, ax = plt.subplots(figsize=(10, 8))
    hb = plot_hexbin(size_image.T, vmin=100, vmax=800, title='RF Size', ax=ax)
    plt.colorbar(hb, ax=ax, label='RF Size (μm)')
    plt.title(f'Receptive Field Size - Cell {CELL_ID}')
    plt.text(0.5, 0.01,
             f'Proximal Size: {prox_size_median:.0f} μm, Distal Size: {dist_size_median:.0f} μm, Max ΔV: {max_v-55:.0f} mV',
             transform=plt.gca().transAxes, ha='center', fontsize=10)
    plt.axis('off')
    
    size_fig_path = os.path.join(FIGURES_DIR, f'rf_size_cell_{CELL_ID}.png')
    plt.savefig(size_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved: rf_size_cell_{CELL_ID}.png")
    
    # Save analysis results
    results_file = os.path.join(OUTPUT_DIR, f'cell_{CELL_ID}_rf_analysis.npz')
    np.savez(
        results_file,
        prox_size_median=prox_size_median,
        dist_size_median=dist_size_median,
        prox_com_median=prox_com_median,
        dist_com_median=dist_com_median,
        max_v=max_v - 55,
        com_image=com_image,
        size_image=size_image,
        g_pas_value=G_PAS,
        nav_den=NAV_DENSITY,
        we=WEIGHT,
        roi_cent=roi_cent
    )
    print(f"      Saved: cell_{CELL_ID}_rf_analysis.npz")
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Data:    {OUTPUT_DIR}")
    print(f"\nSummary:")
    print(f"  Cell ID:                  {CELL_ID}")
    print(f"  Parameters:               g_pas={G_PAS}, Nav={NAV_DENSITY}, weight={WEIGHT}")
    print(f"  Proximal RF size (μm):    {prox_size_median:.0f}")
    print(f"  Distal RF size (μm):      {dist_size_median:.0f}")
    print(f"  Max voltage response (mV): {max_v-55:.1f}")


if __name__ == '__main__':
    main()
