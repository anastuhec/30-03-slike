import numpy as np
from numba import njit, prange

def Pi_bubble_tilde(omega, E_mk, E_nk, Gamma, mu_, invt, nodes, weights, eps=1e-5):
    w = omega / Gamma
    e_mk = E_mk / Gamma
    e_nk = E_nk / Gamma

    T = Gamma / invt

    epsilon_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T) / Gamma

    e_min = min(e_mk, e_nk - w, mu_) - epsilon_max
    e_max = max(e_mk, e_nk - w, mu_) + epsilon_max

    mid = 0.5*(e_max + e_min)
    half = 0.5*(e_max - e_min)
    
    res_mn = 0.0 + 0.0*1j
    res_nm = 0.0 + 0.0*1j

    res_w_mn = 0.0 + 0.0*1j
    res_w_nm = 0.0 + 0.0*1j

    invpi = 1. / np.pi

    n_nodes = len(nodes)
    for i in range(n_nodes):
        e = mid + half * nodes[i]
        ew = e + w
        dm = e - e_mk
        dn = e - e_nk
        dmw = dm + w
        dnw = dn + w
        pref = e - mu_ + 0.5*w

        weight = weights[i]

        f = 1. / (np.exp((e - mu_) * invt) + 1.)
        fw = 1. / (np.exp((ew - mu_) * invt) + 1.)

        A_mk = invpi / (dm*dm + 1.)
        A_nk = invpi / (dn*dn + 1.)
        A_mkw = invpi / (dmw*dmw + 1.)
        A_nkw = invpi / (dnw*dnw + 1.)

        Grnw = 1. / (dnw + 1j)
        Grmw = 1. / (dmw + 1j)

        Gam = 1. / (dm - 1j)
        Gan = 1. / (dn - 1j)

        inte_mn_e = A_mk * Grnw * f + A_nkw * Gam * fw
        inte_nm_e = A_nk * Grmw * f + A_mkw * Gan * fw
    
    
        res_mn += - weight * inte_mn_e
        res_nm += - weight * inte_nm_e

        res_w_mn += - pref * weight * inte_mn_e
        res_w_nm += - pref * weight * inte_nm_e

    return half * res_mn / Gamma, half * res_nm / Gamma, half * res_w_mn, half * res_w_nm

@njit(parallel=True)
def Pi_bubble_tilde_long(omega, E_mk, E_nk, Gamma, mu, T, Nepsilon, eps=1e-5):

    epsilon_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T)

    E_min = min(E_mk, E_nk - omega, mu) - epsilon_max
    E_max = max(E_mk, E_nk - omega, mu) + epsilon_max

    invpi = 1. / np.pi

    epsilons = np.linspace(E_min, E_max, Nepsilon)

    res_mn = np.zeros(Nepsilon, dtype=np.complex128)
    res_nm = np.zeros(Nepsilon, dtype=np.complex128)

    res_w_mn = np.zeros(Nepsilon, dtype=np.complex128)
    res_w_nm = np.zeros(Nepsilon, dtype=np.complex128)

    for i in prange(Nepsilon):
        E = epsilons[i]
        Ew = E + omega
        dm = E - E_mk
        dn = E - E_nk
        dmw = dm + omega
        dnw = dn + omega
        pref = E - mu + 0.5*omega

        f = 1. / (np.exp((E - mu)/T) + 1.)
        fw = 1. / (np.exp((Ew - mu)/T) + 1.)

        A_mk = invpi * Gamma / ((E - E_mk)**2 + Gamma**2)
        A_nk = invpi * Gamma / ((E - E_nk)**2 + Gamma**2)
        A_mkw = invpi * Gamma / (dmw*dmw + Gamma**2)
        A_nkw = invpi * Gamma / (dnw*dnw + Gamma**2)

        Grnw = 1. / (E + omega - E_nk + 1j*Gamma)
        Grmw = 1. / (E + omega - E_mk + 1j*Gamma)

        Gam = 1. / (E - E_mk - 1j * Gamma)
        Gan = 1. / (E - E_nk - 1j * Gamma)

        inte_mn_e = A_mk * Grnw * f + A_nkw * Gam * fw
        inte_nm_e = A_nk * Grmw * f + A_mkw * Gan * fw
    
        res_mn[i] = - inte_mn_e
        res_nm[i] = - inte_nm_e

        res_w_mn[i] = - pref * inte_mn_e
        res_w_nm[i] = - pref * inte_nm_e

    deps = epsilons[1] - epsilons[0]
    res_mn_out = np.sum(res_mn) * deps
    res_nm_out = np.sum(res_nm) * deps
    res_w_mn_out = np.sum(res_w_mn) * deps
    res_w_nm_out = np.sum(res_w_nm) * deps
    return res_mn_out, res_nm_out, res_w_mn_out, res_w_nm_out