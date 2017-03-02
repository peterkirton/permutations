# permutations

Code for time-evolving and finding steady states of permutation symmetric quantum mechanical problems

Hard Requirements:  
python 2.7  
modern versions of scipy and numpy  

Soft requirements:  
qutip (for wigner functions)  
pylab (for plotting)  

An example is given in run_Dicke.py

This sets up the problem studied in arXiv 1611.03342 of the dissipative Dicke model.
It then does an example of time evolution and steady state finding.

To define a new model add to models.py. The examples there show both the dissipative Dicke model and a simple laser model.

