"""
base_functions.py - Core functions for RGC receptive field simulations

This module contains functions for:
- Loading SWC morphology files into NEURON
- Distributing synapses uniformly along dendrites
- Creating stimulus grids for receptive field mapping
- Running NEURON simulations with Nav channels
- Extracting voltage responses from simulations

Author: Pratyush Ramakrishna
Date: 2025
"""

from __future__ import division
from neuron import h

# Load NEURON standard libraries
h.load_file('stdrun.hoc')
h.load_file('stdlib.hoc')
h.load_file('import3d.hoc')

import numpy as np


class Cell:
    """
    A simple container for NEURON cell morphology.
    
    Attributes
    ----------
    soma : list
        List of soma sections
    apic : list
        List of apical dendrite sections
    dend : list
        List of basal dendrite sections
    axon : list
        List of axon sections
    all : list
        Combined list of all sections
    """
    
    def __init__(self, name='neuron', soma=None, apic=None, dend=None, axon=None):
        self.name = name
        self.soma = soma if soma is not None else []
        self.apic = apic if apic is not None else []
        self.dend = dend if dend is not None else []
        self.axon = axon if axon is not None else []
        self.all = self.soma + self.apic + self.dend + self.axon

    def delete(self):
        """Clear all section references."""
        self.soma = None
        self.apic = None
        self.dend = None
        self.axon = None
        self.all = None

    def __str__(self):
        return self.name


def load_swc(filename, fileformat=None, cell=None, use_axon=True, 
             xshift=0, yshift=0, zshift=0):
    """
    Load a morphology from an SWC file into NEURON.
    
    Parameters
    ----------
    filename : str
        Path to the SWC file
    fileformat : str, optional
        File format ('swc' or 'asc'). Auto-detected if None.
    cell : Cell, optional
        Existing Cell object to populate. Creates new if None.
    use_axon : bool
        Whether to include axon sections
    xshift, yshift, zshift : float
        Coordinate offsets to apply to all points
        
    Returns
    -------
    cell : Cell
        Cell object containing all sections
    real_secs : dict
        Dictionary mapping section names to Section objects
    """
    if cell is None:
        cell = Cell(name="".join(filename.split('.')[:-1]))

    if fileformat is None:
        fileformat = filename.split('.')[-1]

    name_form = {1: 'soma[%d]', 2: 'axon[%d]', 3: 'dend[%d]', 4: 'apic[%d]'}

    if fileformat == 'swc':
        morph = h.Import3d_SWC_read()
    elif fileformat == 'asc':
        morph = h.Import3d_Neurolucida3()
    else:
        raise Exception('file format `%s` not recognized' % (fileformat))
    morph.input(filename)

    i3d = h.Import3d_GUI(morph, 0)

    swc_secs = i3d.swc.sections
    swc_secs = [swc_secs.object(i) for i in range(int(swc_secs.count()))]

    sec_list = {1: cell.soma, 2: cell.axon, 3: cell.dend, 4: cell.apic}
    real_secs = {}

    for swc_sec in swc_secs:
        cell_part = int(swc_sec.type)
        if (not use_axon and cell_part == 2) or swc_sec.is_subsidiary:
            continue
        if cell_part not in name_form:
            raise Exception('unsupported point type')
        name = name_form[cell_part] % len(sec_list[cell_part])
        sec = h.Section(cell=cell)
        if swc_sec.parentsec is not None:
            sec.connect(real_secs[swc_sec.parentsec.hname()](swc_sec.parentx))
        if swc_sec.first == 1:
            h.pt3dstyle(1, swc_sec.raw.getval(0, 0), swc_sec.raw.getval(1, 0),
                        swc_sec.raw.getval(2, 0), sec=sec)
        j = swc_sec.first
        xx, yy, zz = [swc_sec.raw.getrow(i).c(j) for i in range(3)]
        dd = swc_sec.d.c(j)
        if swc_sec.iscontour_:
            raise Exception('Unsupported section style: contour')
        if dd.size() == 1:
            x, y, z, d = [dim.x[0] for dim in [xx, yy, zz, dd]]
            for xprime in [x - d / 2., x, x + d / 2.]:
                h.pt3dadd(xprime + xshift, y + yshift, z + zshift, d, sec=sec)
        else:
            for x, y, z, d in zip(xx, yy, zz, dd):
                h.pt3dadd(x + xshift, y + yshift, z + zshift, d, sec=sec)

        sec_list[cell_part].append(sec)
        real_secs[swc_sec.hname()] = sec

        # Set the number of segments based on the section length
        sec.nseg = max(1, int(sec.L / 1))

    cell.all = cell.soma + cell.apic + cell.dend + cell.axon
    return cell, real_secs


def xyz_at(sec, v):
    """
    Get interpolated (x, y) coordinates at normalized position v along section.
    
    Parameters
    ----------
    sec : h.Section
        NEURON section
    v : float
        Normalized position along section (0-1)
        
    Returns
    -------
    x, y : float
        Coordinates at position v
    """
    n = int(h.n3d(sec=sec))
    ar = np.array([h.arc3d(i, sec=sec) for i in range(n)])
    x = np.array([h.x3d(i, sec=sec) for i in range(n)])
    y = np.array([h.y3d(i, sec=sec) for i in range(n)])
    d = v * ar[-1]
    j = np.searchsorted(ar, d)
    if j == 0:
        return x[0], y[0]
    f = (d - ar[j-1]) / (ar[j] - ar[j-1])
    return x[j-1] + f * (x[j] - x[j-1]), y[j-1] + f * (y[j] - y[j-1])


def distribute_synapses_uniform_perturbed(cell, density=0.06, noise_scale=0.5, seed=None):
    """
    Like distribute_synapses_uniform but add a small random perturbation to each
    normalized position along a section.

    Parameters
    ----------
    cell : Cell
        Cell object with .all sections.
    density : float
        Synapse density (same meaning as distribute_synapses_uniform).
    noise_scale : float in [0, 1]
        Fraction of the allowed maximum perturbation. The allowed maximum is
        half the distance between neighboring (evenly spaced) positions, so
        noise_scale=1.0 permits perturbations up to that half-distance.
        Typical values: 0.0 (no noise) .. 1.0 (max allowed).
    seed : int or None
        Optional RNG seed for reproducibility.

    Returns
    -------
    secs, synapses, syn_x, syn_y
        Same format as distribute_synapses_uniform. synapses contain the
        perturbed normalized positions.
    """
    rng = np.random.default_rng(seed)
    secs     = []
    synapses = []
    syn_x    = []
    syn_y    = []

    for sec in cell.all:
        lam = sec.L * density
        n_syn = int(np.ceil(lam))
        sec_pos, sec_x, sec_y = [], [], []
        if n_syn > 0:
            # base even spacing (same as original)
            vals = np.linspace(0, 1, n_syn + 2)[1:-1]          # even spacing
            # spacing between neighbors in normalized coords
            spacing = 1.0 / (n_syn + 1)
            # maximum perturbation allowed to keep points from crossing neighbors:
            max_perturb = 0.5 * spacing
            # scale by user parameter
            max_perturb *= float(np.clip(noise_scale, 0.0, 1.0))
            if max_perturb > 0:
                # draw independent normal perturbations, then clip to [-max_perturb, max_perturb]
                # scale chosen so most draws fall well within the bound; clipping guarantees the hard limit.
                scale = max_perturb * 0.5
                perturb = rng.standard_normal(size=vals.shape) * scale
                perturb = np.clip(perturb, -max_perturb, max_perturb)
                vals = vals + perturb
                vals = np.clip(vals, 1e-12, 1.0 - 1e-12)

            for v in vals:
                sec_pos.append(float(v))
                x, y = xyz_at(sec, v)
                sec_x.append(x); sec_y.append(y)

        secs.append(sec)
        synapses.append(sec_pos)
        syn_x.append(sec_x)
        syn_y.append(sec_y)

    return secs, synapses, syn_x, syn_y

def grid_synapses_pruned(syn_x_off, syn_y_off, secs_off, grid_size=50, region_size=100):
    """
    Create a stimulus grid and find which grid cells contain synapses.
    
    Parameters
    ----------
    syn_x_off, syn_y_off : list of arrays
        X, Y coordinates of synapses for each section
    secs_off : list
        List of sections
    grid_size : float
        Size of grid cells (distance between adjacent grid centers, μm)
    region_size : float
        Size of region to check for synapses around each grid point (μm)
        
    Returns
    -------
    x_po : ndarray
        X coordinates of all grid cell centers
    y_po : ndarray
        Y coordinates of all grid cell centers
    po_list : ndarray
        Indices of grid cells that contain synapses
    syn_masks_stim : dict
        Dictionary mapping grid indices to synapse masks
    """
    x_po = []
    y_po = []
    po_list = []
    syn_masks_stim = {}
    po = 0

    # Create grid covering 1400 x 1400 μm area
    for i in range(0, 1400, grid_size):
        for j in range(0, 1400, grid_size):
            syn_mask_this_round = []
            has_synapses = False

            for k in range(len(secs_off)):
                if len(syn_x_off[k]) > 0.5:
                    syn_mask = np.zeros(len(syn_x_off[k]), dtype=bool)
                    
                    for l in range(len(syn_x_off[k])):
                        half_size = region_size / 2
                        syn_mask[l] = ((i - half_size) < syn_x_off[k][l] < (i + half_size)) and \
                                     ((j - half_size) < syn_y_off[k][l] < (j + half_size))
                        if syn_mask[l]:
                            has_synapses = True
                else:
                    syn_mask = []

                syn_mask_this_round.append(syn_mask)

            x_po.append(i)
            y_po.append(j)
            syn_masks_stim[po] = syn_mask_this_round

            if has_synapses:
                po_list.append(po)

            po += 1

    x_po = np.array(x_po)
    y_po = np.array(y_po)
    po_list = np.unique(po_list)

    return x_po, y_po, po_list, syn_masks_stim


def find_closest_section(sections, x, y, z, threshold=0.1):
    """
    Find the section closest to a given (x, y, z) coordinate.
    
    Parameters
    ----------
    sections : dict
        Dictionary of section names to Section objects
    x, y, z : float
        Target coordinates (μm)
    threshold : float
        Early termination threshold (μm)
        
    Returns
    -------
    closest_section : h.Section
        The section closest to the target point
    """
    min_distance = float('inf')
    closest_section = None

    for sec in sections.values():
        n3d = int(h.n3d(sec=sec))
        x3d = np.array([h.x3d(i, sec=sec) for i in range(n3d)])
        y3d = np.array([h.y3d(i, sec=sec) for i in range(n3d)])
        z3d = np.array([h.z3d(i, sec=sec) for i in range(n3d)])

        distances = np.sqrt((x3d - x)**2 + (y3d - y)**2 + (z3d - z)**2)
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        if min_dist < min_distance:
            min_distance = min_dist
            closest_section = sec

        if min_distance <= threshold:
            break

    return closest_section


def run_simulation_square(swc_file, g_pas, ppp, we, syn_masks_stim_off,
                          syn_x_off, syn_y_off, synapses_off, nav_density,
                          x_pos, y_pos, convolved_frame):
    """
    Run a NEURON simulation with synaptic input at a specified grid position.
    
    Parameters
    ----------
    swc_file : str
        Path to the SWC morphology file
    g_pas : float
        Passive leak conductance (S/cm²)
    ppp : int
        Grid position index
    we : float
        Synaptic weight scaling factor
    syn_masks_stim_off : dict
        Synapse masks for each grid position
    syn_x_off, syn_y_off : list of lists
        Synapse coordinates
    synapses_off : list of lists
        Normalized synapse positions on sections
    nav_density : float
        Nav channel density (S/cm²)
    x_pos, y_pos : ndarray
        Grid position coordinates
    convolved_frame : ndarray
        Stimulus intensity pattern
        
    Returns
    -------
    volt_list : list of tuples
        List of (segment, voltage_vector) for every segment
    """
    # Load morphology
    cell1, secs1 = load_swc(swc_file)

    # Insert mechanisms into every section
    for sec in cell1.all:
        sec.insert('pas')
        sec.Ra = 0.1
        sec.g_pas = g_pas
        sec.e_pas = -55.0
        sec.cm = 1
        sec.insert('NaV_pr')
        sec.gMax_NaV_pr = nav_density
        sec.ena = 50

    # Instantiate synapses
    instantiated_synapses = []
    syn_weight_instantiated = []

    for key, syn_mask_this_round in syn_masks_stim_off.items():
        i, j = x_pos[key], y_pos[key]
        if i == x_pos[ppp] and j == y_pos[ppp]:
            for k, syn_mask in enumerate(syn_mask_this_round):
                if len(syn_mask) > 0:
                    xcc = i
                    ycc = j
                    for l in range(len(syn_x_off[k])):
                        if syn_mask[l]:
                            synapse_value = synapses_off[k][l]
                            x_off = int(xcc - syn_x_off[k][l] + 50)
                            y_off = int(ycc - syn_y_off[k][l] + 50)
                            weight = convolved_frame[x_off, y_off]
                            closest_sec = find_closest_section(
                                secs1, syn_x_off[k][l], syn_y_off[k][l], 25
                            )
                            syn = h.Exp2Syn(synapse_value, sec=closest_sec)
                            syn.e = 0
                            syn.tau1 = 1
                            syn.tau2 = 1
                            instantiated_synapses.append(syn)
                            syn_weight_instantiated.append(weight)

    # Create stimulus
    stim = h.NetStim()
    stim.number = 1
    stim.start = 0.2
    stim.interval = 1
    stim.noise = 0

    # Connect synapses to stimulus
    netcons = []
    for idx, syn in enumerate(instantiated_synapses):
        nc = h.NetCon(stim, syn)
        nc.weight[0] = we * 2 * abs(syn_weight_instantiated[idx] - 0.5)
        netcons.append(nc)

    # Record voltages
    volt_list = []
    for sec in cell1.all:
        for seg in sec:
            v_vec = h.Vector().record(seg._ref_v)
            volt_list.append((seg, v_vec))

    # Run simulation
    h.dt = 0.01
    h.finitialize(-55)
    h.continuerun(3)

    return volt_list


def max_voltage_positions(volt_list, xpos, ypos, distance_threshold=1000, decimals=2):
    """
    Extract maximum voltage at each segment and its spatial coordinates.
    
    Parameters
    ----------
    volt_list : list of tuples
        List of (segment, voltage_vector) pairs
    xpos, ypos : float
        Center coordinates for distance filtering (μm)
    distance_threshold : float
        Maximum distance from center to include (μm)
    decimals : int
        Decimal places for rounding
        
    Returns
    -------
    x_coords : ndarray
        X coordinates of segments (μm)
    y_coords : ndarray
        Y coordinates of segments (μm)
    max_voltage_values : ndarray
        Maximum voltages (mV)
    """
    # Compute max voltages
    volt_array = np.array([v_vec.to_python() for seg, v_vec in volt_list])
    max_volt_values = np.max(volt_array, axis=1)
    max_voltages = [(seg, max_v) for (seg, _), max_v in zip(volt_list, max_volt_values)]

    x_coords = []
    y_coords = []
    max_voltage_values = []
    section_data = {}

    for seg, max_v in max_voltages:
        sec = seg.sec
        seg_x = seg.x

        if sec not in section_data:
            n3d = int(h.n3d(sec=sec))
            x3d = np.array([h.x3d(i, sec=sec) for i in range(n3d)])
            y3d = np.array([h.y3d(i, sec=sec) for i in range(n3d)])
            z3d = np.array([h.z3d(i, sec=sec) for i in range(n3d)])
            diffs = np.sqrt(np.diff(x3d)**2 + np.diff(y3d)**2 + np.diff(z3d)**2)
            cum_lengths = np.insert(np.cumsum(diffs), 0, 0.0)
            total_length = cum_lengths[-1]
            section_data[sec] = (x3d, y3d, z3d, diffs, cum_lengths, total_length)

        x3d, y3d, z3d, lengths, cum_lengths, total_length = section_data[sec]
        target_length = seg_x * sec.L

        if target_length <= total_length:
            idx = np.searchsorted(cum_lengths, target_length) - 1
            if idx < 0:
                idx = 0

            seg_len = lengths[idx]
            if seg_len == 0:
                frac = 0.0
            else:
                frac = (target_length - cum_lengths[idx]) / seg_len

            x_pt = x3d[idx] + frac * (x3d[idx + 1] - x3d[idx])
            y_pt = y3d[idx] + frac * (y3d[idx + 1] - y3d[idx])

            if np.hypot(x_pt - xpos, y_pt - ypos) < distance_threshold:
                x_coords.append(x_pt)
                y_coords.append(y_pt)
                max_voltage_values.append(max_v)

    return (np.round(np.array(x_coords), decimals),
            np.round(np.array(y_coords), decimals),
            np.round(np.array(max_voltage_values), decimals))
