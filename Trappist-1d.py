#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import R_earth, M_earth, G, k_B, M_sun
from scipy.interpolate import interp1d


# Same Constants For all Plots 

m_H = (3.34 * 10**(-27)) * (u.kg)
magnetic_p = 4 * np.pi * 1e-7  * (u.H / u.m) 


T_exo_H = 900  * (u.K)
R_D = 0.788 * R_earth.to(u.m)
rd_exo = R_D + (396000 * (u.m))
M_D = 0.388 * M_earth.to(u.kg)

nd_sw = 1.20e4  *(1/u.cm**3)
vd_sw = 566000  * (u.m/u.s)                              
prot_mass = 1.67e-27 * u.kg

Omega_pc_E = 0.63
r_exo_E = 6871000 * u.m
rd_IMB = 7647000  * u.m          # Used Earth IMB value
formfact_o = 1.16
Mag_Earth = 8e22    * (u.A * u.m**2)  
mass = (1.67*10**(-27)) * (u.kg)


magnetic_moments = np.logspace(17, 27, 500)   * (u.A * u.m**2)  
magnetic_m_values_earth_units = magnetic_moments / Mag_Earth



# Converting Constant
earth_atmosphere_in_kg = (1e-6 * M_earth)  
seconds_per_gyr = (1e9 * 365.25 * 24 * 3600)



# Gunell Pickup Escape

Q0_pu_H = 5e26  * (1/u.s)
hd_H = (k_B * T_exo_H * rd_exo**2) / (G * M_D * m_H)

PU_escape = []
pickup_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nd_sw * vd_sw**2 * prot_mass))**(1/6)
    if R_MP <= rd_exo:
        Q_pu = Q0_pu_H
    else:
        num = 2*hd_H**3 + 2*hd_H**2*R_MP + hd_H*R_MP**2
        den = 2*hd_H**3 + 2*hd_H**2*rd_exo + hd_H*rd_exo**2
        exp_term = np.exp((rd_exo - R_MP) / hd_H)
        Q_pu = Q0_pu_H * (num / den) * exp_term
    pickup_vals.append(Q_pu)

for Q_pu in pickup_vals:
    mass_loss_rate =  Q_pu * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    PU_escape.append(mass_loss_rate_atmospheres.value)  
PU_escape = u.Quantity(PU_escape)




# Cross-Field Ion Loss

Q0_cf_H = 7.7e25    * (1/u.s)

CF_escape = []
cf_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nd_sw * vd_sw**2 * prot_mass))**(1/6)
    if R_MP >= rd_IMB:
        temp = 1 - (rd_exo / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cf = Q0_cf_H * ((1 - (Omega_pc / (4 * np.pi))) / (1 - (Omega_pc_E / (4 * np.pi))))
    cf_vals.append(Q_cf)

for Q_cf in cf_vals:
    mass_loss_rate =  Q_cf * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    CF_escape.append(mass_loss_rate_atmospheres.value)

CF_escape = u.Quantity(CF_escape)


# Polar Cap Loss

Q0_pc_H = 7.8e25   * (1/ u.s)

PC_escape = []
pc_vals = []
for M in magnetic_moments:
    R_MP = (((magnetic_p * formfact_o**2 * M**2) / (8 * np.pi**2 * nd_sw * vd_sw**2 * prot_mass))**(1/6))
    if R_MP > rd_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_pc = Q0_pc_H * (Omega_pc / Omega_pc_E) * ((rd_exo / r_exo_E)**2)
    pc_vals.append(Q_pc)

for Q_pc in pc_vals:
    mass_loss_rate =  Q_pc * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    PC_escape.append(mass_loss_rate_atmospheres.value)  

PC_escape = u.Quantity(PC_escape)


# Cusp Escape

Q0_cu_H = 5e24     *( 1/u.s)           # Earth Params
Qmax_cu_H = 5e25   *( 1/u.s)         # Earth Params


# Calculate r_c_E for scaling
nsw = (1e7) * (1 / u.m**3)  
vsw = (604000) * (u.m / u.s)   
r_c_E = ((magnetic_p * Mag_Earth**2) / (8 * np.pi**2 * nsw * vsw**2 * prot_mass))**(1/6)

CU_escape = []
cusp_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nd_sw * vd_sw**2 * prot_mass))**(1/6)
    r_c_d = R_MP
    if R_MP > rd_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cu = min(Q0_cu_H * (r_c_d / r_c_E)**2, Qmax_cu_H) * (Omega_pc / Omega_pc_E) * (rd_exo / r_exo_E)**2
    cusp_vals.append(Q_cu)

for Q_cu in cusp_vals:
    mass_loss_rate =  Q_cu * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    CU_escape.append(mass_loss_rate_atmospheres.value)

CU_escape = u.Quantity(CU_escape)


# Calculate Peter's (Ion Pickup Escape)

MPU_escape = []
magnetic_limited_vals = []

time = 10  * (u.s)
mass_water = 1e21 * (u.kg)
sigma_coll = (1 * 10**(-17))  * (u.m**2)
energy_eff = (1 / (10.6e44))

for M in magnetic_moments:
    r_mp = (((magnetic_p * M**2) / (8 * np.pi**2 * nd_sw * vd_sw**2 * prot_mass))**(1/6))  
    h_exo_d = (k_B * T_exo_H * rd_exo**2) / (G * M_D * m_H)
    n_exo_d = (1 / (h_exo_d * sigma_coll))  #.to(1 / u.m**3)
    n_L_d = n_exo_d * np.exp((-rd_exo / h_exo_d) * (1 - (rd_exo / r_mp)))
    surface_a_d = (4 * np.pi * r_mp**2)
    f_lm_d = (((energy_eff * h_exo_d * surface_a_d * (n_L_d - nd_sw)) / time) * mass_water) #.decompose()
    magnetic_limited_vals.append(f_lm_d)

for f_lm_d in magnetic_limited_vals:
    mass_loss_rate =  f_lm_d * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    MPU_escape.append(mass_loss_rate_atmospheres.value)

MPU_escape =u.Quantity(MPU_escape)



# Calculate energy-limited escape rate

a = (40.216 * 84e6) * (u.m)     # Used ratio from Eric Agol paper ( a/R*)
M_star = 0.09*M_sun
epsilon = 0.1  # heating efficiency

# Stellar XUV Luminosity
L_bol = 3.828e26 * u.W       # solar luminosity
L_xuv = 3.4e-4 * L_bol       
F_xuv = L_xuv / (4 * np.pi * a**2)                      # flux at planet

# Its labeled as the mass density of the plnet. I used the formula to calculate density: mass/V  
rho = M_D / ((4/3) * np.pi * R_D**3)                    

# Roche lobe radius (Becker+2020 uses 2 in denominator!)
r_roche = a * (M_D / (2 * M_star))**(1/3)

# Tidal enhancement factor K
K = 1 - (3/2) * (R_D / r_roche) + (1/2) * (R_D / r_roche)**3

# Energy-limited escape rate (mass loss) in kg/s
mdot = epsilon * (R_D / R_D)**2 * (3 * F_xuv) / (4 * G * rho * K)
mdot = mdot.to(u.kg / u.s)

seconds_per_gyr = (1e9 * 365.25 * 24 * 3600) * u.s
mdot_earth_per_gyr = (mdot * seconds_per_gyr / M_earth)


mdot_array = u.Quantity(np.full(magnetic_moments.shape, mdot_earth_per_gyr.value), unit=mdot_earth_per_gyr.unit)



# Total Escape  
total_escape = PU_escape.value + CU_escape.value + PC_escape.value + CF_escape.value


plt.figure(figsize=(10, 7))

plt.loglog(magnetic_m_values_earth_units, mdot_array, label='Energy Limited Escape', color='red', linewidth=3)
plt.loglog(magnetic_m_values_earth_units,  MPU_escape, label = 'Driscoll Ion Pickup Escape', color = 'yellow')
plt.loglog(magnetic_m_values_earth_units, PU_escape, '--', label='Pickup H Escape', color='orange')
plt.loglog(magnetic_m_values_earth_units, CF_escape, '-', label='Cross-field H Escape', color='blue')
plt.loglog(magnetic_m_values_earth_units, PC_escape, label='Polar Cap H Escape', color='purple')
plt.loglog(magnetic_m_values_earth_units, CU_escape, '--', label='Cusp H Escape', color='hotpink')
plt.loglog(magnetic_m_values_earth_units, total_escape, linewidth=2.5, color='black', label='Total H Escape')


plt.xlabel('Magnetic Dipole Moment [A·m²]')
plt.ylabel('Earth atm / Gyr')
plt.title('TRAPPIST-1d Total Hydrogen Escape vs Magnetic Moment')
plt.legend()
plt.ylim(1e-35, 1e5)
plt.xlim(1e-4, 1e2)
plt.show()


# In[ ]:




