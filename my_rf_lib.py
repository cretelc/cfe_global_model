import numpy as np
import matplotlib.pyplot as plt
import cmath 


class Impedance():

    def self():
        return 
    

def calculate_impedance(R, C, L, f):
    omega = 2 * np.pi * f
    return complex(R, (omega*L - (1/(omega*C))))

def parallel_impedance(*args):
    Zeq_inv = 0
    for Z in args:
        Zeq_inv = Zeq_inv + (1/Z)
    return 1/Zeq_inv

def series_impedance(*args):
    Zeq_inv = 0
    for Z in args:
        Zeq_inv += Z
    return Zeq_inv




if __name__ == '__main__':
    # System resistance, ohms
    R = 2
    # System inductance, H
    L = 1.86e-9 
    # System capacitance, F
    C = 258e-12
    # Applied frequency, Hz
    f = 13.56e6

    Z_test = calculate_impedance(R, C, L, f)
    print(Z_test)