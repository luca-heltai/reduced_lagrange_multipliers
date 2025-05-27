#!/Users/heltai/anaconda3/bin/python3
"""
Utility script to convert blood vessel network .net files to VTK format
for use with the deal.II blood flow application.

Usage:
    python convert_net_to_vtk.py input.net output.vtk

The .net format is:
# node1 node2 length inlet_radius outlet_radius wave_speed inlet_bc_type outlet_bc_type resistance1 resistance2 compliance
"""

import numpy as np
import pyvista as pv
import sys
import os


def read_net_file(filename):
    """Read vessel network data from .net file"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                values = line.split()
                if len(values) >= 11:  # Ensure we have all required fields
                    data.append([float(x) for x in values])

    return data


def write_vtk_file(filename, vessel_data):
    """Write vessel network as VTK file with line segments using PyVista"""
    n_vessels = len(vessel_data)

    # Create points (nodes) - collect all unique nodes
    nodes = set()
    for vessel in vessel_data:
        nodes.add(int(vessel[0]))  # inlet node
        nodes.add(int(vessel[1]))  # outlet node

    nodes = sorted(list(nodes))
    n_nodes = len(nodes)
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # Create points array - place nodes along x-axis for simplicity
    # In a real application, you would have actual 3D coordinates
    points = np.zeros((n_nodes, 3))
    x_pos = 0.0

    for i, node in enumerate(nodes):
        points[i] = [x_pos, 0.0, 0.0]
        if i < len(nodes) - 1:
            # Estimate distance to next node based on vessel lengths
            connected_lengths = []
            for vessel in vessel_data:
                if int(vessel[0]) == node or int(vessel[1]) == node:
                    connected_lengths.append(vessel[2])  # length

            if connected_lengths:
                x_pos += sum(connected_lengths) / len(connected_lengths)
            else:
                x_pos += 1.0

    # Create cells array for VTK_LINE elements
    cells = []
    for vessel in vessel_data:
        inlet_idx = node_to_index[int(vessel[0])]
        outlet_idx = node_to_index[int(vessel[1])]
        # 2 = number of points per line
        cells.extend([2, inlet_idx, outlet_idx])

    cells = np.array(cells)

    # Create the unstructured grid
    grid = pv.UnstructuredGrid(cells, np.full(
        n_vessels, pv.CellType.LINE), points)

    # Add cell data (vessel properties)
    grid.cell_data["vessel_id"] = np.arange(n_vessels, dtype=float)
    grid.cell_data["length"] = np.array([vessel[2] for vessel in vessel_data])
    grid.cell_data["inlet_radius"] = np.array(
        [vessel[3] for vessel in vessel_data])
    grid.cell_data["outlet_radius"] = np.array(
        [vessel[4] for vessel in vessel_data])
    grid.cell_data["wave_speed"] = np.array(
        [vessel[5] for vessel in vessel_data])
    grid.cell_data["inlet_bc_type"] = np.array(
        [vessel[6] for vessel in vessel_data])
    grid.cell_data["outlet_bc_type"] = np.array(
        [vessel[7] for vessel in vessel_data])
    grid.cell_data["resistance1"] = np.array(
        [vessel[8] for vessel in vessel_data])
    grid.cell_data["resistance2"] = np.array(
        [vessel[9] for vessel in vessel_data])
    grid.cell_data["compliance"] = np.array(
        [vessel[10] for vessel in vessel_data])

    # Save to VTK file
    grid.save(filename, binary=False)
    print(f"Saved {n_vessels} vessels and {n_nodes} nodes to {filename}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_net_to_vtk.py input.net output.vtk")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    print(f"Reading vessel network from {input_file}")
    vessel_data = read_net_file(input_file)

    if len(vessel_data) == 0:
        print("Error: No vessel data found in input file")
        sys.exit(1)

    print(f"Found {len(vessel_data)} vessels")
    print(f"Writing VTK file to {output_file}")

    write_vtk_file(output_file, vessel_data)
    print("Conversion complete!")


if __name__ == "__main__":
    main()
