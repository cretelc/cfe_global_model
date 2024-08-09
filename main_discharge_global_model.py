import numpy as np
from scipy.constants import electron_mass as me
from scipy.constants import elementary_charge as q
from scipy.constants import Avogadro as N_A
from scipy.constants import Boltzmann as kB
from scipy.constants import speed_of_light as c
from scipy.constants import epsilon_0 as eps0
from scipy.special import jv as bessel 
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import global_model_lib as gm

import matplotlib.pyplot as plt

#rint(kB)
#print(unit('Boltzmann'))
mu_0 = 1.25663706127e-6

def sccm2mgs(sccm, Ma):
    return 7.43583e-4 * Ma * sccm
def sccm2atoms(sccm):
    return 4.477962e17 * sccm 

def dne_dt(ne, ng, Kiz, uB, Aeff, V):
    return 
def dng_dt(Q0,V, n, uB, Aeff1, Kiz, Gamma_g, Ag):
    return

def collision_processes(ne, ng, Te, M, rate_constants, energies ):
    Kiz, Kex, Kel = rate_constants
    Eiz, Eex, Eel = energies
    ionization = ne*ng*Kiz*Eiz
    excitation = ne*ng*Kex*Eex
    elastic_thermal = ne*ng*(3*me/M)*Kel*kB*Te
    return [ionization, excitation, elastic_thermal]

def energy_wall_loss(ne, ng, Te, M, R, L, sig_i):
    uB   = gm.calculate_uB(Te, M)
    Aeff = gm.calculate_Aeff(R, L, sig_i, ng)
    V    = gm.calculate_Vol(R, L)
    wall_power = 7*kB*Te*ne*uB*Aeff/V
    return wall_power

def test_bed_equations(t, vars):
    ne, ng, EFg, EFe= vars
    Te = 2*EFe / (3*ne*kB)
    Tg = 2*EFg / (3*ng*kB)
    # Constants
    Tw      = 400
    Eiz     = 12.127 # V
    Eiz_J   = gm.V2J(Eiz)
    Eexc    = 11.6 # V
    Eexc_J  = gm.V2J(Eexc)
    
    sig_i   = 1e-18
    sig_el  = sig_i
    kappa   = 0.0057
    N       = 5
    R       = 0.06
    L       = 0.1
    f       = 13.56e6
    M       = gm.Au2kg(131)
    Q0      = 1.2e19
    beta_i  = 0.7
    #beta_i = Bi
    beta_g  = 0.3
    #beta_g  = Bg
    Icoil   = 5
    # Calculated Parameters
    Kiz   = gm.calculate_Kiz(Te, Eiz)
    Kel   = 1e-13
    Kin   = gm.calculate_Kin(Tg, M, sig_i)
    Kexc  = gm.calculate_Kex(Te, Eexc) # only shows up in Ploss equation
    V     = gm.calculate_Vol(R, L)
    Aeff  = gm.calculate_Aeff(R, L, sig_i, ng)
    Aeff1 = gm.calculate_Aeff1(R, L, ng, sig_i, beta_i) # not a constant
    A     = gm.calculate_A(R, L)
    Ag    = gm.calculate_Ag(beta_g, R)
    uB    = gm.calculate_uB(Te, M)
    vg    = gm.calculate_v(Tg, M)
    Gam_g = gm.calculate_Gam_g(ng, vg)
    Lam0  = gm.calculate_Lam0(R, L)

    # Breakdown of the ne diff. eq.
    #dne_dt_in = ne*ng*Kiz
    #dne_dt_out = ne*uB*Aeff/V

    # Breakdown of the Te diff. eq.
    k0   = gm.calculate_k0(f)
    wpe  = gm.calculate_wpe(ne)
    ve   = gm.calculate_v(Te, me)
    mu_m = gm.calculate_mu_m(ng, ve, sig_el)
    ep   = gm.calculate_eps_p(wpe, f, mu_m)
    k    = gm.calculate_k(k0, ep)
    Rind = gm.calculate_Rind(L, N, R, f, k, ep)
    Pabs = gm.calculate_Pabs(V, Rind, Icoil)

    # Breakdown of Ploss
    PlossA = Eiz_J * ne * ng * Kiz 
    PlossB = Eexc_J * ne * ng * Kexc 
    PlossC = 3 * (me/M) * kB * (Te-Tg) * ne * ng * Kel
    PlossD = (7*kB*Te) * ne * uB * (Aeff/V)
    Ploss_vec = np.array([PlossA, PlossB, PlossC, PlossD])
    Ploss_per = Ploss_vec / sum(Ploss_vec)
    Ploss  = PlossA + PlossB + PlossC + PlossD

    dne_dt = ne*ng*Kiz - ne*uB*Aeff/V
    dng_dt =  (Q0/V) + ne*uB*Aeff1/V - ne*ng*Kiz - Gam_g*Ag/V
    # neutral power balance equation 
    dEFg_dt = 3 * (me/M)*kB*(Te-Tg)*ne*ng*Kel + (1/4) * M*(uB**2) * ne*ng*Kin - kappa * ((Tg-Tw)/Lam0) * (A/V)

    dEFe_dt = (Pabs - Ploss)
    return [dne_dt, dng_dt, dEFg_dt, dEFe_dt]

def equations(t, vars, Eiz, Eexc, M, N, R, Rc, lc, L, Q0, Bi, Bg, Kel, Tw, kappa, Icoil, f, sig_el, sig_i):
    ne, ng, EFg, EFe= vars
    Te = 2*EFe / (3*ne*kB)
    Tg = 2*EFg / (3*ng*kB)
    Eiz_J   = gm.V2J(Eiz)
    Eexc_J  = gm.V2J(Eexc)
    beta_i = Bi
    beta_g  = Bg
    # Calculated Parameters
    Kiz   = gm.calculate_Kiz(Te, Eiz)
    Kel   = 1e-13
    Kin   = gm.calculate_Kin(Tg, M, sig_i)
    Kexc  = gm.calculate_Kex(Te, Eexc) # only shows up in Ploss equation
    V     = gm.calculate_Vol(R, L)
    Aeff  = gm.calculate_Aeff(R, L, sig_i, ng)
    Aeff1 = gm.calculate_Aeff1(R, L, ng, sig_i, beta_i) # not a constant
    A     = gm.calculate_A(R, L)
    Ag    = gm.calculate_Ag(beta_g, R)
    uB    = gm.calculate_uB(Te, M)
    vg    = gm.calculate_v(Tg, M)
    Gam_g = gm.calculate_Gam_g(ng, vg)
    Lam0  = gm.calculate_Lam0(R, L)

    # Breakdown of the Te diff. eq.
    k0   = gm.calculate_k0(f)
    wpe  = gm.calculate_wpe(ne)
    ve   = gm.calculate_v(Te, me)
    mu_m = gm.calculate_mu_m(ng, ve, sig_el)
    ep   = gm.calculate_eps_p(wpe, f, mu_m)
    k    = gm.calculate_k(k0, ep)
    Rind = gm.calculate_Rind(L, N, R, f, k, ep)
    #Pabs = gm.calculate_Pabs(V, Rind, Icoil)
    Vcoil = 20
    l_cap = lc
    A_cap = 2*np.pi*Rc*l_cap
    C     = 0
    Icoilt = gm.Icoil_t(Icoil, f, t)
    #Icoilt = Icoil
    Pabs_cap = gm.calculate_PabsV3(ne, ng, Te, sig_el, L, N, R, Rc, lc, Icoilt, Vcoil, f, l_cap, A_cap, C )
    Pabs = Pabs_cap / V

    # Breakdown of Ploss
    PlossA = Eiz_J * ne * ng * Kiz 
    PlossB = Eexc_J * ne * ng * Kexc 
    PlossC = 3 * (me/M) * kB * (Te-Tg) * ne * ng * Kel
    PlossD = (7*kB*Te) * ne * uB * (Aeff/V)
    Ploss_vec = np.array([PlossA, PlossB, PlossC, PlossD])
    Ploss_per = Ploss_vec / sum(Ploss_vec)
    Ploss  = PlossA + PlossB + PlossC + PlossD

    dne_dt = ne*ng*Kiz - ne*uB*Aeff/V
    dng_dt =  (Q0/V) + ne*uB*Aeff1/V - ne*ng*Kiz - Gam_g*Ag/V
    # neutral power balance equation 
    dEFg_dt = 3 * (me/M)*kB*(Te-Tg)*ne*ng*Kel + (1/4) * M*(uB**2) * ne*ng*Kin - kappa * ((Tg-Tw)/Lam0) * (A/V)
    dEFe_dt = (Pabs - Ploss)
    return [dne_dt, dng_dt, dEFg_dt, dEFe_dt]

## --- Main function ---
def main():
    t_start = 0 # s
    t_end   = 1e-3 # s

    # Initial guesses
    Te0 = 5     # initial electron temperature, eV
    Tg0 = 100   # initial gas temperature, K
    ne0  = 1e16  # initial gas density, m-3
    ng0 = 1e18  # initial plasma density, m-3

    EFe0 = (3/2) * (ne0*kB*gm.eV2K(Te0))
    EFg0 = (3/2) * (ng0*kB*Tg0)

    # Model and Plasma constants 
    sig_i   = 1e-18 # used as the global variable for elastic and cex cross sections
    sig_el  = sig_i
    kappa   = 0.0057 # xenon thermal conductivity
    Eexc    = 11.6
    Eiz     = 12.127
    Kel     = 1e-13     # m3 s-1 
    Mi      = 131
    M       = gm.Au2kg(Mi)

    # Thruster Geometry 
    R  = 0.06  # Discharge chamber radius, m
    L  = 0.10   # Discharge chamber length, m
    lc = L/2    # Coil length, m
    t  = 0.001   # Chamber thickness, m
    Rc = 0.07  # Coil radius, m
    N  = 5      # number of turns
    Bi = 0.70    # ion transparency
    Bg = 0.30    # neutral gas transparency

    # Operating points 
    mdot    = 1.25           # Particle flow rate, sccm 
    Q0      = 1.2e19     # particle injection rate, s-1
    #Q0      = sccm2atoms(mdot)
    Va      = 1500       # grid potential, V
    lg      = 0.00036    # grid separation, m
    f       = 13.56e6    # applied frequency, Hz
    Icoil   = 5          # current amplitude in the coil, A
    Tw      = 400        # wall temperature, K

    initial_guesses = [gm.eV2K(Te0), Tg0, ne0, ng0]
    params = (Eiz, Eexc, M, N, R, Rc, lc, L, Q0, Bi, Bg, Kel, Tw, kappa, Icoil, f, sig_el, sig_i)
             #Eiz, Eexc, M, N, R, Rc, lc, L, Q0, Bi, Bg, Kel, Tw, kappa, Icoil, f, sig_el, sig_i)
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    meth = 'Radau'
    #sol = solve_ivp(test_bed_equations, [0,0.0005], [ne0, ng0, EFg0, EFe0], method=meth)#, max_step=1/(10*f))
    sol = solve_ivp(equations, [t_start, t_end], [ne0, ng0, EFg0, EFe0], t_eval=np.linspace(t_start, t_end, 1000), args = params, method=meth)#, max_step=1/(10*f))
    ne_sol = sol.y[0]
    ng_sol = sol.y[1]
    Te_sol = 2*sol.y[3] / (3*ne_sol*kB)
    Tg_sol = 2*sol.y[2] / (3*ng_sol*kB)
    # Plot Results
    
    i = 0
    t = sol.t * f
    axs[i].semilogy(t, ng_sol, label=f'ng - {meth}')
    axs[i].semilogy(t, ne_sol, label=f'ne - {meth}')
    axs[i].set_xlabel(r'Time [$ms$]')
    axs[i].set_ylabel(r'Temperature [$eV$]')
    axs[i].grid(which='both')
    axs[i].legend()

    i = 1
    ax_left = axs[i].twinx()
    #axs[i].plot(t, sol.y[0], label=r'$Ionization Fraction$')
    colors = ["tab:blue", "tab:orange"]
    axs[i].plot(t, Te_sol/11606, color=colors[1], label=f'Te - {meth}')
    ax_left.plot(t, Tg_sol, color=colors[0], label=f'Tg - {meth}')
    axs[i].set_xlabel(r'Time [$ms$]')
    axs[i].set_ylabel(r'Temperature [$eV$]', color=colors[1])
    ax_left.set_ylabel(r'Neutral Temperature [$K$]', color=colors[0])
    axs[i].grid(which='both')
    #axs[i].legend()
    plt.show()
    print('Program complete.')
    print('Steady State values:')
    print(f"ne = {sol.y[0][-1]:0.3e}")
    print(f"ng = {sol.y[1][-1]:0.3e}")
    print(f"fiz= {100*sol.y[0][-1]/(sol.y[0][-1] + sol.y[1][-1]):0.2f} %")
    print(f"Tg = {Tg_sol[-1]:0.2f} K")
    print(f"Te = {gm.K2eV(Te_sol[-1]):0.3f} eV")

def test(t_start, t_end, f_MHz, A):
    f = f_MHz * (1e6)
    t = np.linspace(t_start, t_end)
    I = A * np.sin(2*np.pi*f*t)
    Igm = gm.Icoil_t(1, 2*np.pi*f, t)
    plt.plot(t, I)
    plt.plot(t, Igm, label='Igm')
    plt.legend()
    plt.show()
    return 


if __name__ == "__main__":
    main()
    #test(0, 2e-7, 13.56, 1)

