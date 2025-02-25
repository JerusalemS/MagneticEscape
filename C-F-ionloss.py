import numpy as np
import matplotlib.pyplot as plt

# Constants
Q0_cf_H = 7.7e25  # s^-1
r_IMB = 7647e3  # in meters
r_exo = 6871e3  # in meters
n_sw = 6e6  # m^-3
v_sw = 4e5  # m/s
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space in H/m
Omega_pc_E = 0.63  # solid angle for Earth
prot_mass = (1.67e-27) 


# Magnetic dipole moments range
magnetic_moments = np.logspace(16, 28, 100)

# Lists to store results
R_mp_values = []
Omega_pc_values = []
escape_rate_values = []

# Calculating values
for magnetic_moment in magnetic_moments:
    # Calculate magnetopause radius
    R_mp = ((mu_0 * magnetic_moment**2) / 
             (8 * (np.pi**2) * n_sw * (v_sw**2)* prot_mass))**(1/6)

    R_mp_values.append(R_mp)

    # Check if R_mp is calculated correctly
    print(f"Magnetic Moment: {magnetic_moment:.2e}, R_mp: {R_mp:.2e}")

    # Calculate solid angle Omega_pc
    if r_IMB <= R_mp:
        temp_value = 1 - (r_exo / R_mp)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp_value)) if temp_value >= 0 else 0
    else:
        Omega_pc = 0  

    Omega_pc_values.append(Omega_pc)
    # Calculate escape rate
    escape_rate = Q0_cf_H * ((1 - Omega_pc / (4 * np.pi))/(1 - Omega_pc_E / (4 * np.pi)))
    escape_rate_values.append(escape_rate)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(magnetic_moments, escape_rate_values, label='Qcf,H', color='blue')
plt.xscale('log')  # Optional: log scale for x-axis
plt.yscale('log')  # Optional: log scale for y-axis
plt.title('Cross-Field Ion Loss Escape Rates vs. Magnetic Dipole Moments')
plt.xlabel('Magnetic Dipole Moment (m_dp) [kg m^3/s^2]')
plt.ylabel('Escape Rate (Qcf,H) [s^-1]')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
#check