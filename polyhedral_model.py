# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 15:38:22 2026

@author: tewki
"""

import sys
print(sys.executable)
import os
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads/Summer Research/Data_Apophis")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from polyhedral_gravity import Polyhedron, evaluate, PolyhedronIntegrity, NormalOrientation, GravityEvaluable, MetricUnit

import analytical_cube_gravity
import mesh_plotting
import time

start = time.time()

#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

# pull files for faces and vertices
file_path1 = os.path.join(downloads_path, "shape_f.dat")
file_path2 = os.path.join(downloads_path, "shape_v.dat")

# import Tsoulis data for comparison
file_path3 = os.path.join(downloads_path, "pot_tsulis.dat")
tsoulis_data = np.loadtxt(f'{file_path3}')
tsoulis_xyz = tsoulis_data[:,:3]
tsoulis_potential = tsoulis_data[:,3]

faces = np.loadtxt(f'{file_path1}')
faces = faces.astype(int) - 1
vertices = np.loadtxt(f'{file_path2}')

density = 1.75e12  # [kg/km^3]

apophis_polyhedron = Polyhedron(
    polyhedral_source=[vertices, faces],
    density=density,
    normal_orientation=NormalOrientation.OUTWARDS,
    integrity_check=PolyhedronIntegrity.DISABLE,
    metric_unit=MetricUnit.KILOMETER,
    )

computation_points = tsoulis_xyz  # shape (1002001, 3)
evaluable_apophis = GravityEvaluable(apophis_polyhedron)

batch_size = 100_000
num_points = tsoulis_xyz.shape[0]

all_potentials = []
all_accelerations = []
all_tensors = []

for start in range(0, num_points, batch_size):
    end = min(start + batch_size, num_points)
    batch_points = tsoulis_xyz[start:end]
    print(f"Evaluating points {start} to {end}...")

    results_list = evaluable_apophis(batch_points, parallel=True)
    
    if isinstance(results_list, list):
        all_potentials.append(np.concatenate([np.atleast_1d(r[0]) for r in results_list]))
        all_accelerations.append(np.vstack([np.atleast_2d(r[1]) for r in results_list]))
        all_tensors.append(np.vstack([np.atleast_2d(r[2]) for r in results_list]))
    else:
        all_potentials.append(np.atleast_1d(results_list.potential))
        all_accelerations.append(np.atleast_2d(results_list.acceleration))
        all_tensors.append(np.atleast_2d(results_list.tensor))

potential = np.concatenate(all_potentials)
acceleration = np.vstack(all_accelerations)
tensor = np.vstack(all_tensors)

end = time.time()
print("Runtime:", end - start, "seconds")

print("Done!")




