# warp2tensor
try to convert back and forth between warp and tensor

a h5 file containing a numpy array is loaded, then data is transferred between warp and torch

Run python conversion_test.py

The result I got is 
Warp initialized:
   Version: 0.4.3
   Devices:
     "cpu"    | x86_64
     "cuda:0" | NVIDIA GeForce RTX 3090
   Kernel cache: /home/multiples/.cache/warp/0.4.3
warm start: 2.072s
  10001x warm start cost: 1.230s (59%)
  10001x actual cost 2nd try torch2warp: 0.341s (16%)
  10001x actual cost 1st try torch2warp: 0.331s (15%)
  10001x actual cost 2nd try warp2torch: 0.085s (4%)
  10001x actual cost 1st try warp2torch: 0.085s (4%)
  
  # Why is torch2warp much slower than warp2torch?
