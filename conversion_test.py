import warp as wp
import warp.torch
from timer_cm import Timer
import numpy as np
import argparse
import h5py
import torch


h5file = h5py.File("./double_cone_highRes_initial_sampling.h5", 'r')
particle_q_np = h5file['q']
particle_q_np = np.transpose(particle_q_np) # n by 3 numpy array


device = "cuda"
wp.init()

particle_q = wp.from_numpy(particle_q_np, dtype=wp.vec3, device=device) # initialize warp array

timer = Timer("warm start")
num_wm = 10001
for i in range(num_wm):
    with timer.child('warm start cost'):
        particle_q_t = wp.to_torch(particle_q) # warp array to tensor
        particle_q = wp.from_torch(particle_q_t) # tensor to warp array
        assert(str(particle_q_t.data_ptr()) == str(particle_q.ptr)) # this is to ensure the correct pointer is transferred
for i in range(num_wm):
    with timer.child('actual cost 1st try torch2warp'):
        particle_q = wp.from_torch(particle_q_t)
    with timer.child('actual cost 1st try warp2torch'):
        particle_q_t = wp.to_torch(particle_q)
for i in range(num_wm):
    with timer.child('actual cost 2nd try warp2torch'):
        particle_q_t = wp.to_torch(particle_q)
    with timer.child('actual cost 2nd try torch2warp'):
        particle_q = wp.from_torch(particle_q_t)
    # the first try and the second try have different ordering, so ordering is not the problem
timer.print_results()    
