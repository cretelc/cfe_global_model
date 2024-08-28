import numpy as np
from scipy.constants import electron_mass as me
from scipy.constants import elementary_charge as q
from scipy.constants import Avogadro as N_A
from scipy.constants import Boltzmann as kB
from scipy.constants import speed_of_light as c
from scipy.constants import epsilon_0 as eps0
from scipy.special import jv as bessel 
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import fsolve
import global_model_lib as gm
from time import perf_counter
import performance_lib as gm_perf

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

    # Geometry -- verified, should be used as an input
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
    #mu_m  = gm.calculate_mu_m(ng, ve, sig_el)
    mu_m = gm.calculate_mu_m(ng, Kel)
    ep    = gm.calculate_eps_p(wpe, f, mu_m)
    k     = gm.calculate_k(k0, ep)
    Rind  = gm.calculate_Rind(L, N, R, f, k, ep)
    l_cap = lc
    A_cap = 2*np.pi*R*l_cap
    C     = 10e-12 # 2e-11 is what Chris calculated
    #Icoilt = gm.calculate_Icoil(Icoil, 2*np.pi*f, t)
    Icoilt = Icoil
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
    l_cap = lc/2 # 0.05 m
    A_cap = 2*np.pi*R*l_cap
    C     = 1e-11
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
def run_model():

    # Start and End Time
    t_start = 0 # s
    t_end   = 1e-3 # s

    # Initial guesses
    # Electron Tempearture (eV)
    Te0 = 2.6     
    # neutral Temperature (K)
    Tg0 = 300   
    # Electron density (m-3)
    ne0 = 5e17  
    # Neutral Density (m-3)
    ng0 = 1e19 
    # Electron energy flux
    EFe0 = (3/2) * (ne0*kB*gm.eV2K(Te0))
    # Neutral energy flux
    EFg0 = (3/2) * (ng0*kB*Tg0)

    ## Plasma constants
    # Global cross section (CEX, elastic)  
    sig_i   = 1e-18 
    # Elastic collision cross section
    sig_el  = 2e-19
    # Gas thermal conductivity 
    kappa   = 0.0057 # xenon thermal conductivity
    # Excitation Energy (Xe)
    Eexc    = 11.6
    # Ionization Energy (Xe)
    Eiz     = 12.127
    # Elastic collision rate, m3 s-1  
    Kel     = 1e-13      
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
    Rc = 0.07  
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
    Tw      = 323        


    # Validate constant variables 
    # Geometry -- verified, should be used as an input
    print("--- Geometry ---")
    V     = gm.calculate_Vol(R, L)
    print(f"Plasma Volume: {V:0.3e} m-3")
    A     = gm.calculate_A(R, L)
    print(f"Total Area: {A:0.3e} m2")
    Ag    = gm.calculate_Ag(Bg, R)
    print(f"Grid Area: {Ag:0.3e} m2")
    A_cap = 2*np.pi*R*(lc)
    print(f"Cap Area: {A_cap:0.3e} m2")
    Lam0  = gm.calculate_Lam0(R, L)
    print(f"Heat Diffusion Length: {Lam0:0.3e} m")

    print("--- Electrical ---")
    Lcoil = gm.calculate_Lcoil(Rc, N, lc)
    print(f'Coil Inductance: {Lcoil:0.3e} H')

    print("--- Operating Inputs ---")
    print(f"Propellant Mass: {M:0.3e} kg")
    print(f"Neutral Gas Velocity: {gm.calculate_v(Tg0, M):0.3f} m/s")


    params = (Eiz, Eexc, M, N, R, Rc, lc, L, Q0, Bi, Bg, Kel, Tw, kappa, Icoil, f, sig_el, sig_i)
    meth = 'Radau'
    T = 1/(2*np.pi*f)

    density_fig, density_ax_L = plt.subplots(2, 3, figsize=(12,6))
    density_ax_L = density_ax_L.flat

    temperature_fig, temperature_ax_L = plt.subplots(2, 3, figsize=(12,6))
    temperature_ax_L = temperature_ax_L.flat

    ms = T/2
    Icoil = [41.3, 30.9, 25.4, 22.9, 20.3, 18.9] # from Mansur's analysis
    runtimes = np.zeros(len(Icoil))
    #Is = np.logspace(-1, 2)
    #results = np.zeros((4, len(Is)))
    for j,I in enumerate(Icoil):
        print(f'Running with Icoil = {I} A ')
        params = (Eiz, Eexc, M, N, R, Rc, lc, L, Q0, Bi, Bg, Kel, Tw, kappa, I, f, sig_el, sig_i)
        t0 = perf_counter()
        
        sol_CC = solve_ivp(equations,    [t_start, t_end], [ne0, ng0, EFg0, EFe0], t_eval=np.linspace(t_start, t_end, 500), args = params, method=meth, max_step=ms)
        runtimes[j] = perf_counter() - t0
        print(f"Runtime: {perf_counter()-t0:0.3f}s")
        print('---')

        # Plot Results
        
        ne_sol = sol_CC.y[0]
        ng_sol = sol_CC.y[1]
        Te_sol = 2*sol_CC.y[3] / (3*ne_sol*kB)
        Tg_sol = 2*sol_CC.y[2] / (3*ng_sol*kB)
        t = sol_CC.t 
        
        colors = ["tab:blue", "tab:orange"]
        density_ax_R = density_ax_L[j].twinx()
        density_ax_L[j].plot(t*(1e3), ng_sol*(1e-18), color=colors[1], linewidth=1)
        density_ax_R.plot(t*(1e3),    ne_sol*(1e-17), color=colors[0], linewidth=1)
        density_ax_L[j].set_xlabel(r'Time [$ms$]')
        density_ax_L[j].set_ylabel(r'$n_g$ [$10^{18} m^{-3}$]', color=colors[0])
        density_ax_R.set_ylabel(r'$n_e$ [$10^{17} m^{-3}$]', color=colors[1])
        density_ax_R.set_title(f'Const. Icoil={I}')
        density_ax_L[j].grid(which='both')


        temperature_ax_R = temperature_ax_L[j].twinx()
        temperature_ax_R.plot(t*(1e3), Te_sol/11606, color=colors[1], linewidth=1)
        temperature_ax_L[j].plot(t*(1e3), Tg_sol, linewidth=1, color=colors[0])
        temperature_ax_L[j].set_xlabel(r'Time [$ms$]')
        temperature_ax_R.set_ylabel(r'$T_e$ [$eV$]', color=colors[1])
        temperature_ax_L[j].set_ylabel(r'$T_g$ [$K$]', color=colors[0])
        temperature_ax_R.set_title(f'Const. Icoil={I}')
        temperature_ax_L[j].grid(which='both')
        
    density_fig.tight_layout()
    temperature_fig.tight_layout()
    plt.show()

def compare_to_chabert():

    # Start and End Time
    t_start = 0 # s
    t_end   = 1e-3 # s

    # Initial guesses
    # Electron Tempearture (eV)
    Te0 = 2.6     
    # neutral Temperature (K)
    Tg0 = 300   
    # Electron density (m-3)
    ne0 = 5e17  
    # Neutral Density (m-3)
    ng0 = 1e19 
    # Electron energy flux
    EFe0 = (3/2) * (ne0*kB*gm.eV2K(Te0))
    # Neutral energy flux
    EFg0 = (3/2) * (ng0*kB*Tg0)

    ## Plasma constants
    # Global cross section (CEX, elastic)  
    sig_i   = 1e-18 
    # Elastic collision cross section
    sig_el  = 2e-19
    # Gas thermal conductivity 
    kappa   = 0.0057 # xenon thermal conductivity
    # Excitation Energy (Xe)
    Eexc    = 11.6
    # Ionization Energy (Xe)
    Eiz     = 12.127
    # Elastic collision rate, m3 s-1  
    Kel     = 1e-13      
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
    Rc = 0.07  
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
    Tw      = 323        

    # solve_ivp inputs
    params = (Eiz, Eexc, M, N, R, Rc, lc, L, Q0, Bi, Bg, Kel, Tw, kappa, Icoil, f, sig_el, sig_i)
    use_method = 'Radau'
    T = 1/(2*np.pi*f)
    ms = T/2
    # Parameter sweep
    Icoil = [41.3, 30.9]#, 25.4, 22.9, 20.3, 18.9] # from Mansur's analysis
    runtimes = np.zeros(len(Icoil))
    #Is = np.logspace(-1, 2)
    #results = np.zeros((4, len(Is)))
    for j, I in enumerate(Icoil):
        print(f'Running with Icoil = {I} A ')
        params = (Eiz, Eexc, M, N, R, Rc, lc, L, Q0, Bi, Bg, Kel, Tw, kappa, I, f, sig_el, sig_i)
        init_vars = [ne0, ng0, EFg0, EFe0]
        teval = np.linspace(t_start, t_end, 500)
        t0 = perf_counter()
        
        sol_CC = solve_ivp(equations, [t_start, t_end], init_vars, t_eval=teval, args=params, method=use_method, max_step=ms)
        runtimes[j] = perf_counter() - t0
        print(f"Runtime: {perf_counter()-t0:0.3f}s")
        
        # Gather Results
        ne_sol = sol_CC.y[0]
        ng_sol = sol_CC.y[1]
        Te_sol = 2*sol_CC.y[3] / (3*ne_sol*kB)
        Tg_sol = 2*sol_CC.y[2] / (3*ng_sol*kB)
        t = sol_CC.t

        # Steady state values 
        ne_ss = ne_sol[-1]
        ng_ss = ng_sol[-1] 
        Te_ss = Te_sol[-1]
        Tg_ss = Tg_sol[-1]

        # RF Power
        mu_m  = gm.calculate_mu_m(ng_ss, Kel)
        wpe   = gm.calculate_wpe(ne_ss)
        ep    = gm.calculate_eps_p(wpe, f, mu_m)
        k0    = gm.calculate_k0(f)
        k     = gm.calculate_k(k0, ep)
        Rind  = gm.calculate_Rind(L, N, R, f, k, ep)
        Rcoil = 2
        Prf   = gm_perf.rf_pwr(Rind, Rcoil, I)
        print(f"RF Power: {Prf}")

        # Ion thrust
        hL = gm.calc_hL(L, sig_i, ng_ss)
        uB = gm.calculate_uB(Te_ss, M)
        v_beam = gm_perf.beam_velocity(Va, M)
        Ai = gm.calculate_Ai(Bi, R)
        Gam_i = gm_perf.grid_ion_flux(hL, uB)
        Ti = gm_perf.ion_thrust(Gam_i, M, v_beam, Ai)

        # Neutral thrust
        vg = gm.calculate_v(Tg_ss, M)
        Ag = gm.calculate_Ag(Bg, R)
        Gam_g = gm_perf.grid_neutral_flux(ng_ss, vg)
        TN = gm_perf.neutral_thrust(Gam_g, M, vg, Ag)

        print('---')
        
    


if __name__ == "__main__":
    compare_to_chabert()
    #test(0, 2e-7, 13.56, 1)

