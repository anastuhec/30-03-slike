import numpy as np
from pathlib import Path
import sys
sys.path.append("/Users/ana/Desktop/takarada/")
from helpers_takarada import * 
from module_takarada import *

''' numerical parameters for the self-consistent calculation '''
parameters1 = {'n_pass' : 1e-4,
'epsilon_threshold' : 1e-6,
'N_epsilon' : 5,
'maxiter' : 500,
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

params = {'a' : 1.0,
        'b' : 0.2,
        'delta' : 0.0,
        't' : 3.0,
        't_' : 1.0,
        't12' : 0.3,
        'Vb' : 3.0,
        'Vc' : 1.0,
        'Nk' : 2500,
        'include_hartree' : True,
        'mu0' : 0.0,
        'epsilon' : 2.0,
        'epsilon_' : 2.0
        }
 
b = params['b']
t = params['t']
t_ = params['t_']
t12 = params['t12']
Vb = params['Vb']
Vc = params['Vc']
delta = params['delta']
epsilon = params['epsilon']
epsilon_ = params['epsilon_']
phys_parameters = [b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta]

Nk = params['Nk']
mu0 = params['mu0']
include_hartree = params['include_hartree']

''' 1. create excitonic model '''
m = model(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree)
m.GS()

''' 2. heat up a bit '''
beta0 = 50
scale = 1.005
betas = beta0/scale**np.arange(1,3)
stops = [int(np.emath.logn(scale, beta0/beta)) for beta in betas]
Ts = 1/betas
maxbrentq = 20

# set parameters below to random values, because I will not compute transport coefficients in this temperature sweep
Nomega, eps, Gammas = 1000, 1e-5, [0.01]

m.run_Tdependence(betas, stops, Gammas, Nomega, eps, maxbrentq, evaluate_transport=False)

''' 3. simulate pulses'''
A0 = 0.003
t0 = 2.0
sigma = 0.01
Omega0 = 0.0
dt = 0.004
t_max = 500
Ncorr = 100
tol = 1e-5
Gamma_ = 0.0

''' 3.1 simulate pulz in y(delta=0)-direction for excitonic susceptibility '''
rhos = m.rhos
perturbation_operator = rhos[2]
measure_provider = [rhos[0], rhos[1], rhos[2], rhos[3],]

do_freeze = True
time, measurement, norma, delta_bs, delta_cs = simulate_pulz(m.K, m.hk0, m.rho, m.phys_parameters, m.include_hartree, perturbation_operator, measure_provider,
                                                             A0, t0, sigma, Omega0, dt, t_max, do_freeze, Ncorr, tol, m.geom, m.phases, m.g_ffts, Gamma_)
pulz = A_pulz(time, A0, t0, sigma, Omega0)
Results_y = {'time' : time,
             'measurement' : measurement,
             'pulz' : pulz
             }

do_freeze = False
timeN, measurementN, normaN, delta_bsN, delta_csN = simulate_pulz(m.K, m.hk0, m.rho, m.phys_parameters, m.include_hartree, perturbation_operator, measure_provider,
                                                             A0, t0, sigma, Omega0, dt, t_max, do_freeze, Ncorr, tol, m.geom, m.phases, m.g_ffts, Gamma_)
Results_yN = {'time' : timeN,
             'measurement' : measurementN,
             'pulz' : pulz
             }

''' 3.2 simulate pulz in x(delta=0)-direction for excitonic susceptibility '''
perturbation_operator = rhos[1]
measure_provider = [rhos[0], rhos[1], rhos[2], rhos[3],]

do_freeze = True
time, measurement, norma, delta_bs, delta_cs = simulate_pulz(m.K, m.hk0, m.rho, m.phys_parameters, m.include_hartree, perturbation_operator, measure_provider,
                                                             A0, t0, sigma, Omega0, dt, t_max, do_freeze, Ncorr, tol, m.geom, m.phases, m.g_ffts, Gamma_)
Results_x = {'time' : time,
             'measurement' : measurement,
             'pulz' : pulz
             }

do_freeze = False
timeN, measurementN, normaN, delta_bsN, delta_csN = simulate_pulz(m.K, m.hk0, m.rho, m.phys_parameters, m.include_hartree, perturbation_operator, measure_provider,
                                                             A0, t0, sigma, Omega0, dt, t_max, do_freeze, Ncorr, tol, m.geom, m.phases, m.g_ffts, Gamma_)
Results_xN = {'time' : timeN,
             'measurement' : measurementN,
             'pulz' : pulz
             }

''' 3.3. simulate pulz with current '''
perturbation_operator = m.tok
measure_provider = m.tok

do_freeze = True
timeJ, measurementJ, normaJ, delta_bsJ, delta_csJ = simulate_pulz(m.K, m.hk0, m.rho, m.phys_parameters, m.include_hartree, perturbation_operator, measure_provider,
                                                             A0, t0, sigma, Omega0, dt, t_max, do_freeze, Ncorr, tol, m.geom, m.phases, m.g_ffts, Gamma_)
Results_j = {'time' : timeJ,
             'measurement' : measurementJ,
             'pulz' : pulz
             }

do_freeze = False
timeNJ, measurementNJ, normaNJ, delta_bsNJ, delta_csNJ = simulate_pulz(m.K, m.hk0, m.rho, m.phys_parameters, m.include_hartree, perturbation_operator, measure_provider,
                                                             A0, t0, sigma, Omega0, dt, t_max, do_freeze, Ncorr, tol, m.geom, m.phases, m.g_ffts, Gamma_)
Results_jN = {'time' : timeNJ,
             'measurement' : measurementNJ,
             'pulz' : pulz
             }

''' 4. compute excitonic susceptibitiliy via RPA '''

''' 4.1 precompute bubbles '''
Gamma = 0.008
L = 1000
deg = 5000
nodes, weights = np.polynomial.legendre.leggauss(deg)
omegas = np.linspace(0.01,5.0,500)
Pi_w, Pi_eps_w = precompute_bubbles(m.energije, Gamma, m.mu, m.Ts[-1], L, nodes, weights, omegas)

''' 4.2 compute susceptibilities '''
rhos_tilde = operator_tilde(m.rhos, m.vecs)
tok_tilde = operator_tilde(m.tok, m.vecs)
m3, m6, m4a, m4b = ht.compute_all_mf_matrices(m.K, m.rho, m.geom, m.phases, m.g_ffts)
mat = m3 + m6 + m4a + m4b
mat_tilde = operator_tilde(mat, m.vecs)

results = chi_jrho2(Pi_w, Pi_eps_w, rhos_tilde, m.thetas, tok_tilde, mat_tilde)
chi0 = results['chi0']
chi = results['chi']

chi_jj0 = results['chi_jj0']
dchi_jj = results['dchi_jj']
chi_jj = chi_jj0 + dchi_jj

chi_jEj0 = results['chi_jEj0']
dchi_jEj = results['dchi_jEj']
chi_jEj = chi_jEj0 + dchi_jEj

chi_matj0 = results['chi_matj0']
dchi_matj = results['dchi_matj']
chi_matj = chi_matj0 + dchi_matj

Results_chi = {'omegas' : omegas,
               'Gamma' : Gamma,
               
               'chi0' : chi0,
               'chi' : chi,
               
               'chi_jj0' : chi_jj0,
               'chi_jj' : chi_jj,

               'chi_jEj0' : chi_jEj0,
               'chi_jEj' : chi_jEj,
               
               'chi_matj0' : chi_matj0,
               'chi_matj' : chi_matj
               }


Results_all = {'x' : Results_x,
               'xN' : Results_xN,
               'y' : Results_y,
               'yN' : Results_yN,
               'j' : Results_j,
               'jN' : Results_jN,
               'phys_parameters' : m.phys_parameters,
               'chi' : Results_chi
               }

("/Users/ana/Desktop/takarada/")
DATA_DIR = Path( "/Users/ana/Desktop/takarada/lips_data/31-03")
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.savez(DATA_DIR / 'results.npz', **Results_all)