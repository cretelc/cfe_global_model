import json 
import pandas as pd

json_db = pd.DataFrame()


geometry = {
    'chamber_radius'       : 0.06,  # Discharge chamber radius, m
    'chamber_length'       : 0.10,   # Discharge chamber length, m
    'coil_length'          : 0.05,    # Coil length, m
    'chamber_thickness'    : 0.001,   # Chamber thickness, m
    'coil_radius'          : 0.07,  # Coil radius, m
    'coil_turns'           : 5,      # number of turns
    'ion_transparency'     : 0.70,    # ion transparency
    'neutral_transparency' : 0.30    # neutral gas transparency
}

model_inputs = {
    't_start': 0,
    't_end'  : 10e-3,
    't_num'  : 1000 
}

initial_vals = {
    'electron_temperature': 2.5, 
    'neutral_temperature' : 375,  
    'electron_density'    : 5e16,  
    'neutral_density'     : 5e19  
}

plasma_vals = {
    'ion_neutral_x_sect'     : 1e-18, # used as the global variable for elastic and cex cross sections
    'elastic_coll_x_sect'    : 1e-18,
    'thermal_conductivity'   : 0.0057, # xenon thermal conductivity
    'excitation_energy'      : 11.6,
    'ionization_energy'      : 12.127,
    'elastic_collision_rate' : 1e-13,     # m3 s-1 
    'propellant_mass'        : 131
}


operating_conditions = {
# Operating points 
    'flow_rate'            : 1.25,       # Particle flow rate, sccm 
    'accel_grid_potential' : 1500,       # grid potential, V
    'grid_separation'      : 0.00036,    # grid separation, m
    'applied_frequency'    : 13.56e6,    # applied frequency, Hz
    'coil_current'         : 5,          # current amplitude in the coil, A
    'wall_temperature'     : 400         # wall temperature, K
}