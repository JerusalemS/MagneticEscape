import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

# Constants
Q0_cf_H = 7.7e25
r_IMB = 7647000 *(u.m) 
r_exo = 6871000 *(u.m)
n_sw = 6e6 * (1/u.m**3)
v_sw = 4e5 * (u.m/u.s) 
magnetic_p = (4 * np.pi * (1e-7)) * (u.H / u.m) 
Omega_pc_E = 0.63  
prot_mass = (1.67e-27) * (u.kg)


magnetic_moments = np.logspace(16, 28, 10000) * (u.A * u.m**2)

# Lists to store results
R_mp_values = []
Omega_pc_values = []
escape_rate_values = []

# Calculating values
for magnetic_moment in magnetic_moments:
    R_mp = (((magnetic_p * (magnetic_moment**2)*(1.16**2)) / 
             (8 * (np.pi**2) * n_sw * (v_sw**2)*prot_mass))**(1/6)).decompose()

    R_mp_values.append(R_mp.value)

    # Solid angle Omega_pc
    
    if R_mp >= r_IMB:
        temp_value = max(0, 1 - (r_exo / R_mp))
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp_value))
    else:
        Omega_pc = 0
 

    Omega_pc_values.append(Omega_pc)

    escape_rate = Q0_cf_H * ((1 - (Omega_pc / (4 * np.pi)))/(1 - (Omega_pc_E / (4 * np.pi))))
    escape_rate_values.append(escape_rate)

plt.figure(figsize=(10, 6))
plt.loglog(magnetic_moments, escape_rate_values, label='Qcf,H', color='blue') 
plt.title('Cross-Field Ion Loss Escape Rates vs. Magnetic Dipole Moments')
plt.xlabel('Magnetic Dipole Moment  [Am^2]')
plt.ylabel('Escape Rate (Qcf,H) [s^-1]')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()




