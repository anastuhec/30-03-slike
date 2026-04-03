import numpy as np
from pathlib import Path
import sys
sys.path.append("/scratch/stuhecana/takarada/")
from helpers_takarada_lips import *
from lips_takarada.module_takarada_lips import *

DATA_DIR = Path( "/project/stuhecana/takarada/data/02-04")
DATA_DIR.mkdir(parents=True, exist_ok=True)

''' numerical parameters for the self-consistent calculation '''
parameters1 = {'n_pass' : 1e-3,
'epsilon_threshold' : 1e-6,
'N_epsilon' : 5,
'maxiter' : 200,
'eps_last' : 1e-6,
'dmu' : 0.7,
'mix2' : 0.001,
'mix3' : 1.5,
'faktor1' : 0.1,
'max_trials' : 15,
'eps0' : 0.03,
}

parameters2 = {'n_pass' : 1e-3,
'epsilon_threshold' : 1e-6,
'N_epsilon' : 5,
'maxiter' : 100,
'eps_last' : 1e-6,
'dmu' : 0.7,
'mix2' : 0.001,
'mix3' : 1.5,
'faktor1' : 0.1,
'max_trials' : 15,
'eps0' : 0.03,
}


ts = [1.5 + 0.3*i for i in range(6)]

task_id = int(sys.argv[1])
if task_id >= len(ts):
    raise ValueError(f"task_id {task_id} exceeds number of parameter combinations {len(ts)}")
t = ts[task_id]

''' 1. generate ground state of some excitonic insulator model '''
a = 1.
b = 0.5

t12 = 0.0
delta = 0.0

t_ = 1.0
Vb = 3.0
Vc = 1.0

mu0 = 0.
include_hartree = True

Nk = 500

epsilon = 2.0
epsilon_ = 2.0

phys_parameters = [b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta]
m = model(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree)
m.GS()

print(f'delta_b = {np.round(np.abs(m.delta_b), 3)}', flush=True)
print(f'delta_c = {np.round(np.abs(m.delta_c), 3)}', flush=True)
print(f'gap = {np.round(m.gap, 3)}', flush=True)

''' 2. temperature sweep '''
beta0 = 20 * 1 / m.gap
scale = 1.008
betas = beta0/scale**np.arange(1,401)
stops = [int(np.emath.logn(scale, beta0/beta)) for beta in betas]
Ts = 1/betas
maxbrentq = 20

# set parameters below to random values, because I will not compute transport coefficients in this temperature sweep
Gammas = [0.003,0.005,0.008,0.01]
params = {'Nomega' : 1500, 'eps' : 1e-5, 'L' : 1000, 'deg' : 5000, 'omega0' : 1e-6, 'scale' : 5, 'eps2' : 1e-8, 'max_iter' : 1}

m.run_Tdependence(betas, stops, Gammas, params, maxbrentq, evaluate_transport_DC=True, evaluate_vertex_DC=True)

data = m.collect_data()

np.savez(DATA_DIR / f"02-04-{task_id}.npz", **data)