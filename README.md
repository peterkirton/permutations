# permutations

Code for time-evolving and finding steady states of permutation symmetric quantum mechanical problems

Hard Requirements:  
python 3+
modern versions of scipy and numpy  

Soft requirements:  
qutip (for wigner functions)  
matplotlib (for plotting)  
tqdm  (for progress bar in parallel construction of Liouvillian)

A basic example is given in run_Dicke.py

This sets up the problem studied in arXiv 1611.03342 of the dissipative Dicke model.
It then solves for time evolution and the steady state.

To define a new model add to models.py. Example models are included: dissipative Dicke, pumped Dicke, three level system, laser and counter-laser.
