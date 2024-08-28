from scipy.constants import elementary_charge as q
from scipy.constants import Boltzmann as kB
import numpy as np

def ion_thrust(Gamma_i, M, v_beam, Ai):
    return Gamma_i * M * v_beam * Ai

def ion_thrust_pwr(Gamma_i, M, v_beam, Ai):
    return (1/2) * M * Gamma_i * Ai * v_beam**2

def neutral_thrust(Gamma_g, M, v_g, Ag):
    return Gamma_g * M * v_g * Ag

def neutral_thrust_pwr(Gamma_g, M, v_g, Ag):
    return (1/2) * M * Gamma_g * Ag * v_g**2

def grid_ion_flux(hL, n, uB):
    return hL * n * uB

def beam_velocity(Vgrid, M):
    return (2 * q * Vgrid / M) ** (1/2)

def grid_neutral_flux(ng, vg):
    return (1/4) * ng * vg

def neutral_mean_velocity(Tg, M):
    return (8 * kB*Tg / (np.pi * M))

def icp_pwr_eff( Rind, Rcoil):
    return Rind / (Rind + Rcoil)

def thrust_pwr_eff(Pi, PN, Prf):
    return (Pi + PN) / (Pi + PN + Prf)

def thrust_eff(Ti, TN, Prf):
    return (Ti + TN) + Prf

def mass_util_eff(Gamma_i, Ai, Q0):
    return Gamma_i * Ai / Q0

def rf_pwr(Rind, Rcoil, Icoil):
    return (1/2) * (Rind + Rcoil) * Icoil**2
