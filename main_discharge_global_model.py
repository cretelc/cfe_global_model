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
from time import perf_counter

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
    sig_el_e = 2e-19 # electorn-xenon average elastic collision
    mu_m = gm.calculate_mu_m(ng, ve, sig_el_e)
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
    # Electron Tempearture (eV)
    Te0 = 5     
    # neutral Temperature (K)
    Tg0 = 100   
    # Electron density (m-3)
    ne0 = 5e17  
    # Neutral Density (m-3)
    ng0 = 1e18 
    # Electron energy flux
    EFe0 = (3/2) * (ne0*kB*gm.eV2K(Te0))
    # Neutral energy flux
    EFg0 = (3/2) * (ng0*kB*Tg0)

    ## Plasma constants
    # Global heavy-species cross section (CEX, elastic)  
    sig_i   = 1e-18 
    # Elastic collision cross section
    sig_el  = sig_i
    # Electron-heavy elastic cross section 
    sig_el_e = 2e-19 
    # Gas thermal conductivity 
    kappa   = 0.0057 # xenon thermal conductivity
    # Excitation Energy (Xe)
    Eexc    = 11.6
    # Ionization Energy (Xe)
    Eiz     = 12.127
    # Elastic collision rate 
    Kel     = 1e-13     # m3 s-1 
    # Propellant molecular mass 
    Mi      = 131
    M       = gm.Au2kg(Mi)

    ## Thruster Geometry
    # Discharge chamber radius, m 
    R  = 0.06  
    # Discharge chamber length, m
    L  = 0.1   
    # Coil length, m
    lc = L     
    # Chamber thickness, m
    t  = 0.001   
    # Coil radius, m
    Rc = 0.08  
    # number of turns
    N  = 5
    # ion transparency
    Bi = 0.70    
    # neutral gas transparency
    Bg = 0.30     

    ## Operating points 
    # Flow rate (SCCM)
    #mdot    = 1.25
    # Flow rate (atoms/s)   
    #Q0      = sccm2atoms(mdot)
    # Flow rate (atoms/s)         
    Q0      = 1.2e19     
    # Grd Potential (V)
    Va      = 1000       
    # Grid Separation (m)
    lg      = 0.001      
    # Applied frequency (MHz)
    f       = 13.56e6    
    # Coil Current Amplitude (A)
    Icoil   = 5         
    # Wall temperature (K)
    Tw      = 400        


    params = (Eiz, Eexc, M, N, R, Rc, lc, L, Q0, Bi, Bg, Kel, Tw, kappa, Icoil, f, sig_el, sig_i)
    meth = 'Radau'
    T = 1/(2*np.pi*f)
    #max_stepsizes = [T, T/3, T/5, T/20]
    #max_stepsizes_str = ['1/f', '1/3f', '1/5f', '1/20f']
    runtimes = [0, 0, 0, 0]
    ls = ['-', '--', '-.', ':']
    fig1, axs1 = plt.subplots(2, 1, figsize=(12,6))
    ax1_left1 = axs1[0].twinx()
    ax1_left2 = axs1[1].twinx()

    ms = T/5
    Is = [25]
    #Is = np.logspace(-1, 2)
    #results = np.zeros((4, len(Is)))
    for j,Icoil in enumerate(Is):
        params = (Eiz, Eexc, M, N, R, Rc, lc, L, Q0, Bi, Bg, Kel, Tw, kappa, Icoil, f, sig_el, sig_i)
        t0 = perf_counter()
        #if j == 3:
        #    sol_CC = solve_ivp(equations,    [t_start, t_end], [ne0, ng0, EFg0, EFe0], t_eval=np.linspace(t_start, t_end, 1000), args = params, method=meth)
        
        sol_CC = solve_ivp(equations,    [t_start, t_end], [ne0, ng0, EFg0, EFe0], t_eval=np.linspace(t_start, t_end, 10000), args = params, method=meth, max_step=ms)
        runtimes[j] = perf_counter() - t0

        # Plot Results        
        ne_sol = sol_CC.y[0]
        ng_sol = sol_CC.y[1]
        Te_sol = 2*sol_CC.y[3] / (3*ne_sol*kB)
        Tg_sol = 2*sol_CC.y[2] / (3*ng_sol*kB)
        t = sol_CC.t 
        
        i = 0
        
        colors = ["tab:blue", "tab:orange"]
        
        ax1_left1.semilogy(t*(1e3), ng_sol,  linestyle=ls[j], color=colors[0], linewidth=1)
        axs1[i].semilogy(t*(1e3),   ne_sol,  linestyle=ls[j], color=colors[1], linewidth=1, label=f'Icoil={Icoil}')
        axs1[i].set_xlabel(r'Time [$ms$]')
        axs1[i].set_ylabel(r'Density [$m^{-3}$]', color=colors[1])
        ax1_left1.set_ylabel(r'Neutral Gas Density [$m^{-3}$]', color=colors[0])
        #axs1[i].grid(which='both')

        i = 1
        colors = ["tab:blue", "tab:orange"]
        axs1[i].plot(t*(1e3), Te_sol/11606, linestyle=ls[j], color=colors[1], linewidth=1, label=f'Icoil={Icoil}')
        ax1_left2.plot(t*(1e3), Tg_sol, linestyle=ls[j], linewidth=1, color=colors[0])

        axs1[i].set_xlabel(r'Time [$ms$]')
        axs1[i].set_ylabel(r'Temperature [$eV$]', color=colors[1])
        ax1_left2.set_ylabel(r'Neutral Temperature [$K$]', color=colors[0])
        
        
        #print('Program complete.')
        #print('Steady State values:')
        #print(f"ne = {ne_sol[-1]:0.3e}")
        #print(f"ng = {ng_sol[-1]:0.3e}")
        #print(f"fiz= {100*ne_sol[-1]/(ne_sol[-1] + ng_sol[-1]):0.2f} %")
        #print(f"Tg = {Tg_sol[-1]:0.2f} K")
        #print(f"Te = {gm.K2eV(Te_sol[-1]):0.3f} eV")
        #plt.show()


        ''''
        i = 0
        t = sol_CC.t  #t = sol.t * f
        axs2[i].semilogy(t*(1e3), ng_sol, linestyle=ls[j], label=f'ng - {meth}')
        axs2[i].semilogy(t*(1e3), ne_sol, linestyle=ls[j], label=f'ne - {meth}')
        axs2[i].set_xlabel(r'Time [$ms$]')
        axs2[i].set_ylabel(r'Density [$m^{-3}$]')
        axs2[i].grid(which='both')
        axs2[i].legend()

        i = 1
        
        #axs[i].plot(t, sol.y[0], label=r'$Ionization Fraction$')
        colors = ["tab:blue", "tab:orange", 'k']
        axs2[i].plot(t*(1e3), Te_sol/11606, linestyle=ls[j], color=colors[1], label=f'Te - {meth}')
        ax2_left.plot(t*(1e3), Tg_sol, linestyle=ls[j], color=colors[0], label=f'Tg - {meth}')

        axs2[i].set_xlabel(r'Time [$ms$]')
        axs2[i].set_ylabel(r'Temperature [$eV$]', color=colors[1])
        ax2_left.set_ylabel(r'Neutral Temperature [$K$]', color=colors[0])
        axs2[i].grid(which='both')
        #axs[i].legend()
        
        i = 2
        #ax2_left = axs2[i].twinx()
        Igm = gm.calculate_Icoil(Icoil, 2*np.pi*f, t)
        axs2[i].plot(t*(1e3), Igm, linestyle=ls[j], color=colors[2], label=f'Icoil - {meth}')
        ax2_left.plot(t*(1e3), Te_sol/11606, linestyle=ls[j], color=colors[1], label=f'Te - {meth}')

        axs2[i].set_xlabel(r'Time [$s$]')
        axs2[i].set_ylabel(r'Coil current [$A$]', color=colors[2])
        ax2_left.set_ylabel(r'Electron Temperature [$eV$]', color=colors[1])
        axs2[i].grid(which='both')
        #axs[i].set_xlim(0,1e-6)
        '''
    #for i,rt in enumerate(runtimes):
    #    print(f"ne0={Is[i]}, runtime={rt:0.5f}")
    print(f" runtime={runtimes[0]:0.5f}")
    axs1[0].grid(which='both')
    axs1[0].legend()
    axs1[1].grid(which='both')
    axs1[1].legend()
    #fig, ax = plt.subplots(1,1)
    #ax.plot()
    plt.show()
    


if __name__ == "__main__":
    main()
    #test(0, 2e-7, 13.56, 1)

