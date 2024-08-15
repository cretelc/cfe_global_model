import numpy as np
from scipy.constants import electron_mass as me
from scipy.constants import elementary_charge as q
from scipy.constants import Avogadro as N_A
from scipy.constants import Boltzmann as kB
from scipy.constants import speed_of_light as c
from scipy.constants import epsilon_0 as eps0
from scipy.special import jv as bessel 

from scipy.optimize import fsolve

#rint(kB)
#print(unit('Boltzmann'))
mu_0 = 1.25663706127e-6

## Need to define inputs 
# m_dot - mass flow rate
# V_g   - grid potential, potential difference between plasma and accel 
# G     - grid geometry vector [R, L]
# B     - grid transparencies [Bg, Bi]

def calc_hL(L, sig_i, ng):
    '''
    Calculates axial density ratio between center and edge
    verified with Mansur: 
    '''
    lam_i = 0.03
    sig_i = 1e-18
    return 0.86 * (3 + (L/(2 * lam_i))) ** (-1/2) 

def calc_hR(R, sig_i, ng):
    ''' Return the radial edge-to-center density ratio 
    verified with Mansur: 
    '''
    sig_i = 1e-18 
    lam_i = 0.03
    return 0.8 * (4 + (R/lam_i))**(-1/2)

# Rate Constants
def calculate_Kex(Te, Eexc):
    '''
    Calculate excitation rates from Chabert 2012
    verified with Mansur: 
    Te: electron temperature, K
    '''
    Q = kB*Te / q # units=Volt
    return (1.2921e-13) * np.exp(-Eexc / Q)

def calculate_Kiz(Te, Eiz):
    '''
    Ionization rates from Chabert 2012, referencing Fundamentals of Electric Propulsion.
    
    Te: electron temperature, K
    verified with Mansur: 
    '''
    Q = kB*Te / q # units=Volt
    # Kiz1
    term11 = (6.73e-15) * (Q**0.5)
    term12 = 3.97 + 0.643*(Q) - 0.0368 * (Q**2)
    term13 = np.exp(-Eiz / Q)
    # Kiz2
    term21 = (6.73e-15) * (Q**0.5)
    term22 = (-1.031e-4)*(Q**2) + 6.386 * np.exp(-Eiz/Q)

    Kiz1 = term11*term12*term13
    Kiz2 = term21*term22
    return (Kiz1 + Kiz2) / 2

def calculate_Kin(Ti, M, sig_i):
    ''''''
    vi = calculate_v(Ti, M) #np.sqrt(8 * kB*Ti / (np.pi * M))
    return sig_i * vi

def calculate_Aeff(R, L, sig_i, ng):
    hR = calc_hR(R, sig_i, ng)
    hL = calc_hL(L, sig_i, ng)
    return 2*hR*np.pi * R * L + 2 * hL*np.pi*R**2

def calculate_Aeff1(R, L, ng, sig_i, beta_i):
    hR = calc_hR(R, sig_i, ng)
    hL = calc_hL(L, sig_i, ng)
    return 2*hR*np.pi * R * L + (2-beta_i) * hL*np.pi*R**2

# Plasma values
def calculate_uB(Te, M):
    ''' 
    Calculate the Bohm velocity 
    verified
    '''
    return (kB*Te / M)**0.5

def calculate_k(k0, eps_p):
    return k0 * eps_p**(1/2)

def calculate_k0(f):
    w = 2*np.pi * f
    return w / c

def calculate_eps_p(wpe, f, mu_m):
    w = 2*np.pi * f
    return 1 - ((wpe**2) / (w*(complex(w,-mu_m))))

def calculate_wpe(n):
    ''' Calculates electron frequency '''
    return np.sqrt((n * q**2) / (me * eps0))

def calculate_mu_m(ng, ve, sig_el):
    ''' Calculates mean collision frequency '''
    return ng * sig_el * ve

def calculate_v(T, m):
    ''' Calculates thermal velocity '''
    return np.sqrt(8*kB*T / (np.pi*m))

# RF Electrical Calculations
def calculate_Icoil(Icoil_mag, w, t):
    return Icoil_mag * np.sin(w*t)

def calculate_Vcoil(f, Lind, Icoil):
    ''' Calculate approx coil voltage in V, pg. 243 PoRF - Mansur'''
    w = 2 * np.pi * f
    Vcoil = w * Lind * Icoil
    return Vcoil

def calculate_Lcoil(Rc, N, lc):
    ''' Calculates the coil inductances'''
    Lcoil =  mu_0 * np.pi * (Rc * N)**2 / lc
    return Lcoil

def calculate_Rind(L, N, R, f, k, eps_p):
    w = 2 * np.pi * f
    kR = k*R
    J0 = bessel(0, kR)
    J1 = bessel(1, kR)
    leading_term = (2 * np.pi * N**2) / (L * w * eps0)
    complex_term = complex(0, kR*J1/(eps_p*J0))
    return leading_term * complex_term.real

def calculate_Rohm(ne, mu_m, l_cap, A_cap):
    Rohm = me * mu_m * l_cap / ((q**2)*ne*A_cap)
    return Rohm

def calculate_Rstoch(ne, Te, ve, A_cap, Vcoil):
    Rstoch = (me * ve / ((q**2) * ne * A_cap)) * (q * abs(Vcoil) / (kB * Te))**(1/2)
    return Rstoch

def calculate_Rcap(Rohm, Rstoch):
    return Rohm + Rstoch

def calculate_Lind(L, N, R, Rc, Lcoil, f, k, eps_p):
    ''' Calculates the inductance of the plasma circuit'''
    w  = 2 * np.pi * f
    kR = k*R
    J0 = bessel(0, kR)
    J1 = bessel(1, kR)
    leading_term = (2 * np.pi * N**2) / (L * (w**2) * eps0)
    complex_term = complex(0, kR*J1/(eps_p*J0))
    return Lcoil * (1 - (R/Rc)**2) + leading_term*complex_term.imag

def calculate_Ploss(vars, M, K_vec, E_vec, uB, Aeff, V):
    n, ng, Te, Tg  = vars
    Kiz, Kexc, Kel = K_vec
    Eiz, Eexc      = E_vec
    return Eiz*n*ng*Kiz + Eexc*n*ng*Kexc + 3*(me/M)*kB*(Te-Tg)*n*ng*Kel + (7*kB*Te)*n*uB*(Aeff/V)

def calculate_PabsV2(V, Rind, Icoil, f, Lind, Rcap, C=0):
    w = 2*np.pi * f
    return (1/2) * (Rind + ((w**2 * Lind * C)**2) * Rcap)*Icoil**2

def calculate_PabsV3(t, ne, ng, Te, sig_el, L, N, R, Rc, lc, Icoil, f, l_cap, A_cap, C):
    w = 2 * np.pi * f
    
    wpe  = calculate_wpe(ne)
    ve   = calculate_v(Te, me)
    mu_m = calculate_mu_m(ng, ve, sig_el)
    ep   = calculate_eps_p(wpe, f, mu_m)
    k0   = calculate_k0(f)
    k    = calculate_k(k0, ep)
    Lcoil = calculate_Lcoil(Rc, N, lc)
    Rind = calculate_Rind(L, N, R, f, k, ep)
    Lind   = calculate_Lind(L, N, R, Rc, Lcoil, f, k, ep)
    Vcoil = calculate_Vcoil(f, Lind, Icoil)
    
    Rohm   = calculate_Rohm(ne, mu_m, l_cap, A_cap)

    Rstoch = calculate_Rstoch(ne, Te, ve, A_cap, Vcoil)
    Rcap   = calculate_Rcap(Rohm, Rstoch)
    Pabs = (1/2) * (Rind + ((w**2 * Lind * C)**2) * Rcap) * Icoil**2 
    Pcap = (1/2) * ((w**2 * Lind * C)**2) * Rcap*(Icoil**2)
    Pind = (1/2) * Rind*(Icoil**2)
    return Pabs , Pind, Pcap, Rstoch, Rohm

                       #1   2,  3,  4,      5, 6, 7, 8,  9, 10,    11 12,    13,    14, 15

def calculate_PabsV3_MT(t, ne, ng, Te, sig_el, L, N, R, Rc, lc, Icoil, f, l_cap, A_cap, C):
    w = 2 * np.pi * f
    
    wpe  = calculate_wpe(ne)
    ve   = calculate_v(Te, me)
    mu_m = calculate_mu_m(ng, ve, sig_el)
    ep   = calculate_eps_p(wpe, f, mu_m)
    k0   = calculate_k0(f)
    k    = calculate_k(k0, ep)
    Lcoil = calculate_Lcoil(Rc, N, lc)
    Rind = calculate_Rind(L, N, R, f, k, ep)
    Lind = calculate_Lind(L, N, R, Rc, Lcoil, f, k, ep)
    Vcoil = calculate_Vcoil(f, Lind, Icoil)
    
    Rohm = calculate_Rohm(ne, mu_m, l_cap, A_cap)

    Rstoch = calculate_Rstoch(ne, Te, ve, A_cap, Vcoil)
    Rcap = calculate_Rcap(Rohm, Rstoch)
    Pabs = (1/2) * (Rind + ((w**2 * Lind * C)**2) * Rcap)*(Icoil**2)
    Pcap = (1/2) * ((w**2 * Lind * C)**2) * Rcap*(Icoil**2)
    Pind = (1/2) * Rind*(Icoil**2)
    # look at Rind, Lind, Rcap
    return Pabs , Pind, Pcap, Rstoch, Rohm


def calculate_Pabs(V, Rind, Icoil):
    ''' Calculate Absorbed power'''
    #Rind = calculate_Rind(L, N, R, f, k, eps_p)
    return (1/(2*V) * Rind * (Icoil**2)) 

def calculate_Pabs_cap():
    return

def calculate_C(A, d, k):
    return k * eps0 * A / d

def calculate_Vol(R, L):
    ''' Calculates the internal volume of the discharge chamber'''
    return np.pi * L * R**2

def calculate_Ag(Bg, R):
    return np.pi * Bg * R**2

def calculate_Gam_g(ng, vg):
    return (1/4) * ng * vg

def calculate_Lam0(R, L):
    ''' Calculate the heat diffusion length. '''
    # verified
    return (R/2.405) + (L/np.pi)

def calculate_A(R, L):
    return 2 * np.pi *R**2 + 2 * np.pi *R*L





# Conversions
def eV2K(eV):
    ''' Return kelvin (K) given electron-volts (eV) '''
    return 11606*eV

def K2eV(K):
    ''' Return electron-volts (eV) given kelvin (K)'''
    return K/11606

def Au2kg(Au):
    ''' Returns kg given atomic mass units'''
    return Au * 1.6605402E-27 

def V2J(V):
    return V*q

def main():
    print('Testing...')

if __name__== '__main__':
    main()