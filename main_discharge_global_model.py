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

mu_0 = 1.25663706127e-6

def sccm2mgs(sccm, Ma):
    return 7.43583e-4 * Ma * sccm
def sccm2atoms(sccm):
    return 4.477962e17 * sccm 

def dne_dt(ne, ng, Kiz, uB, Aeff, V):
    return 
def dng_dt(Q0,V, n, uB, Aeff1, Kiz, Gamma_g, Ag):
    return

def equations(t, vars, Eiz, Eexc, M, N, R, Rc, lc, L, Q0, Bi, Bg, Kel, Tw, kappa, Icoil, f, sig_el, sig_i):
    # Inputs -- Verified
    ne, ng, EFg, EFe = vars
    Te = 2*EFe / (3*ne*kB)
    Tg = 2*EFg / (3*ng*kB)
    Eiz_J   = gm.V2J(Eiz)
    Eexc_J  = gm.V2J(Eexc)
    beta_i  = Bi
    beta_g  = Bg
    
    # Calculated Parameters
    # Rate Constants -- verified
    Kiz   = gm.calculate_Kiz(Te, Eiz)
    Kel   = 1e-13
    Kin   = gm.calculate_Kin(Tg, M, sig_i)
    Kexc  = gm.calculate_Kex(Te, Eexc) # only shows up in Ploss equation

    # Geometry -- verified
    V     = gm.calculate_Vol(R, L)
    Aeff  = gm.calculate_Aeff(R, L, sig_i, ng)
    Aeff1 = gm.calculate_Aeff1(R, L, ng, sig_i, beta_i) # not a constant
    A     = gm.calculate_A(R, L)
    Ag    = gm.calculate_Ag(beta_g, R)

    # Plasma -- verified
    uB    = gm.calculate_uB(Te, M)
    vg    = gm.calculate_v(Tg, M)
    Gam_g = gm.calculate_Gam_g(ng, vg)
    Lam0  = gm.calculate_Lam0(R, L)

    # Breakdown of the Te diff. eq.
    k0    = gm.calculate_k0(f)
    wpe   = gm.calculate_wpe(ne)
    ve    = gm.calculate_v(Te, me)
    mu_m  = gm.calculate_mu_m(ng, ve, sig_el)
    ep    = gm.calculate_eps_p(wpe, f, mu_m)
    k     = gm.calculate_k(k0, ep)
    Rind  = gm.calculate_Rind(L, N, R, f, k, ep)
    l_cap = lc
    A_cap = 2*np.pi*R*l_cap
    C     = 10e-12 # 2e-11 is what Chris calculated
    Icoilt = gm.calculate_Icoil(Icoil, 2*np.pi*f, t)
    Pabs_cap, Pind, Pcap, Rstoch, Rohm = gm.calculate_PabsV3(t, ne, ng, Te, sig_el, L, N, R, Rc, lc, Icoilt, f, l_cap, A_cap, C)  # edited for Mansur's library
    Pabs = Pabs_cap / V
    
    # Breakdown of Ploss
    PlossA    = Eiz_J * ne * ng * Kiz 
    PlossB    = Eexc_J * ne * ng * Kexc 
    PlossC    = 3 * (me/M) * kB * (Te-Tg) * ne * ng * Kel
    PlossD    = (7*kB*Te) * ne * uB * (Aeff/V)
    Ploss     = PlossA + PlossB + PlossC + PlossD

    dne_dt = ne*ng*Kiz - ne*uB*Aeff/V
    dng_dt =  (Q0/V) + ne*uB*Aeff1/V - ne*ng*Kiz - Gam_g*Ag/V
    # neutral power balance equation 
    dEFg_dt = 3 * (me/M)*kB*(Te-Tg)*ne*ng*Kel + (1/4) * M*(uB**2) * ne*ng*Kin - kappa * ((Tg-Tw)/Lam0) * (A/V)
    dEFe_dt = (Pabs - Ploss)
    return [dne_dt, dng_dt, dEFg_dt, dEFe_dt]


def equations_MT(t, vars, Eiz, Eexc, M, N, R, Rc, lc, L, Q0, Bi, Bg, Kel, Tw, kappa, Icoil, f, sig_el, sig_i):
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
    l_cap = lc
    A_cap = 2*np.pi*R*l_cap
    C     = 10e-12
    Icoilt = gm.calculate_Icoil(Icoil, 2*np.pi*f, t)
    Pabs_cap, Pind, Pcap, Rstoch, Rohm = gm.calculate_PabsV3(t, ne, ng, Te, sig_el, L, N, R, Rc, lc, Icoilt, f, l_cap, A_cap, C)
    Pabs = Pabs_cap / V

    # Breakdown of Ploss
    PlossA = Eiz_J * ne * ng * Kiz 
    PlossB = Eexc_J * ne * ng * Kexc 
    PlossC = 3 * (me/M) * kB * (Te-Tg) * ne * ng * Kel
    PlossD = (7*kB*Te) * ne * uB * (Aeff/V)
    Ploss_vec = np.array([PlossA, PlossB, PlossC, PlossD])
    Ploss_per = Ploss_vec / sum(Ploss_vec)
    Ploss     = PlossA + PlossB + PlossC + PlossD

    dne_dt = ne*ng*Kiz - ne*uB*Aeff/V
    dng_dt = (Q0/V) + ne*uB*Aeff1/V - ne*ng*Kiz - Gam_g*Ag/V
    # neutral power balance equation 
    dEFg_dt = 3 * (me/M)*kB*(Te-Tg)*ne*ng*Kel + (1/4) * M*(uB**2) * ne*ng*Kin - kappa * ((Tg-Tw)/Lam0) * (A/V)
    dEFe_dt = (Pabs - Ploss)
    return [dne_dt, dng_dt, dEFg_dt, dEFe_dt]

## --- Main function ---
def main():

    # Start and End Time
    t_start = 0 # s
    t_end   = 1e-4 # s

    # Initial guesses
    Te0 = 5     # initial electron temperature, eV
    Tg0 = 100   # initial gas temperature, K
    ne0 = 1e16  # initial gas density, m-3
    ng0 = 1e18 # initial plasma density, m-3

    EFe0 = (3/2) * (ne0*kB*gm.eV2K(Te0))
    EFg0 = (3/2) * (ng0*kB*Tg0)

    # Plasma constants 
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
    L  = 0.1   # Discharge chamber length, m
    lc = L     # Coil length, m
    t  = 0.001   # Chamber thickness, m
    Rc = 0.07  # Coil radius, m
    N  = 5      # number of turns
    Bi = 0.70    # ion transparency
    Bg = 0.30    # neutral gas transparency

    # Operating points 
    #mdot    = 1.25           # Particle flow rate, sccm 
    Q0      = 1.2e19     # particle injection rate, s-1
    #Q0      = sccm2atoms(mdot)
    Va      = 1500       # grid potential, V
    lg      = 0.00036      # grid separation, m
    f       = 13.56e6    # applied frequency, Hz
    Icoil   = 5         # current amplitude in the coil, A
    Tw      = 300        # wall temperature, K


    params = (Eiz, Eexc, M, N, R, Rc, lc, L, Q0, Bi, Bg, Kel, Tw, kappa, Icoil, f, sig_el, sig_i)
    meth = 'Radau'
    sol_CC = solve_ivp(equations,    [t_start, t_end], [ne0, ng0, EFg0, EFe0], t_eval=np.linspace(t_start, t_end, 1000), args = params, method=meth)
    #sol_MT = solve_ivp(equations_MT, [t_start, t_end], [ne0, ng0, EFg0, EFe0], t_eval=np.linspace(t_start, t_end, 1000), args = params, method=meth)
    '''
    ne_sol_MT = sol_MT.y[0]
    ng_sol_MT = sol_MT.y[1]
    Te_sol_MT = 2*sol_MT.y[3] / (3*ne_sol_MT*kB)
    Tg_sol_MT = 2*sol_MT.y[2] / (3*ng_sol_MT*kB)
    
    ne_sol_CC = sol_CC.y[0]
    ng_sol_CC = sol_CC.y[1]
    Te_sol_CC = 2*sol_CC.y[3] / (3*ne_sol_CC*kB)
    Tg_sol_CC = 2*sol_CC.y[2] / (3*ng_sol_CC*kB)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(100*(ng_sol_MT - ng_sol_CC)/ng_sol_MT)
    axs[1].plot(100*(Tg_sol_MT - Tg_sol_CC)/Tg_sol_MT)
    plt.show()
    '''
    # Plot Results
    
    ne_sol = sol_CC.y[0]
    ng_sol = sol_CC.y[1]
    Te_sol = 2*sol_CC.y[3] / (3*ne_sol*kB)
    Tg_sol = 2*sol_CC.y[2] / (3*ng_sol*kB)
    t = sol_CC.t 
    fig, axs = plt.subplots(2, 1, figsize=(12,6))
    ax_left1 = axs[0].twinx()
    ax_left2 = axs[1].twinx()
    i = 0
    
    colors = ["tab:blue", "tab:orange"]
    
    ax_left1.semilogy(t*(1e3), ng_sol,  color=colors[0], label=f'ng - {meth}')
    axs[i].semilogy(t*(1e3), ne_sol,  color=colors[1], label=f'ne - {meth}')
    axs[i].set_xlabel(r'Time [$ms$]')
    axs[i].set_ylabel(r'Density [$m^{-3}$]', color=colors[1])
    ax_left1.set_ylabel(r'Neutral Gas Density [$m^{-3}$]', color=colors[0])
    axs[i].grid(which='both')

    i = 1
    colors = ["tab:blue", "tab:orange"]
    axs[i].plot(t*(1e3), Te_sol/11606, color=colors[1], label=f'Te - {meth}')
    ax_left2.plot(t*(1e3), Tg_sol, color=colors[0], label=f'Tg - {meth}')

    axs[i].set_xlabel(r'Time [$ms$]')
    axs[i].set_ylabel(r'Temperature [$eV$]', color=colors[1])
    ax_left2.set_ylabel(r'Neutral Temperature [$K$]', color=colors[0])
    axs[i].grid(which='both')
    
    print('Program complete.')
    print('Steady State values:')
    print(f"ne = {ne_sol[-1]:0.3e}")
    print(f"ng = {ng_sol[-1]:0.3e}")
    print(f"fiz= {100*ne_sol[-1]/(ne_sol[-1] + ng_sol[-1]):0.2f} %")
    print(f"Tg = {Tg_sol[-1]:0.2f} K")
    print(f"Te = {gm.K2eV(Te_sol[-1]):0.3f} eV")
    #plt.show()


    fig, axs = plt.subplots(3,1)
    i = 0
    t = sol_CC.t  #t = sol.t * f
    axs[i].semilogy(t*(1e3), ng_sol, label=f'ng - {meth}')
    axs[i].semilogy(t*(1e3), ne_sol, label=f'ne - {meth}')
    axs[i].set_xlabel(r'Time [$ms$]')
    axs[i].set_ylabel(r'Density [$m^{-3}$]')
    axs[i].grid(which='both')
    axs[i].legend()

    i = 1
    ax_left = axs[i].twinx()
    #axs[i].plot(t, sol.y[0], label=r'$Ionization Fraction$')
    colors = ["tab:blue", "tab:orange", 'k']
    axs[i].plot(t*(1e3), Te_sol/11606, color=colors[1], label=f'Te - {meth}')
    ax_left.plot(t*(1e3), Tg_sol, color=colors[0], label=f'Tg - {meth}')
    axs[i].set_xlabel(r'Time [$ms$]')
    axs[i].set_ylabel(r'Temperature [$eV$]', color=colors[1])
    ax_left.set_ylabel(r'Neutral Temperature [$K$]', color=colors[0])
    axs[i].grid(which='both')
    #axs[i].legend()
    
    i = 2
    ax_left = axs[i].twinx()
    Igm = gm.calculate_Icoil(Icoil, 2*np.pi*f, t)
    axs[i].plot(t*(1e3), Igm, color=colors[2], label=f'Icoil - {meth}')
    ax_left.plot(t*(1e3), Te_sol/11606, color=colors[1], label=f'Te - {meth}')
    axs[i].set_xlabel(r'Time [$s$]')
    axs[i].set_ylabel(r'Coil current [$A$]', color=colors[2])
    ax_left.set_ylabel(r'Electron Temperature [$eV$]', color=colors[1])
    axs[i].grid(which='both')
    #axs[i].set_xlim(0,1e-6)
    plt.show()
    


if __name__ == "__main__":

    main()
    #test(0, 2e-7, 13.56, 1)

