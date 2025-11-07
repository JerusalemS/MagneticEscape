#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import R_earth, M_earth, G, k_B, M_sun
from scipy.interpolate import interp1d


# Same Constants For all Plots 

m_H = (3.34 * 10**(-27)) * (u.kg)
magnetic_p = 4 * np.pi * 1e-7  * (u.H / u.m) 


T_exo_H = 900  * (u.K)
R_B = 1.116 * R_earth.to(u.m)
rb_exo = R_B + (396000 * u.m)
M_B = 1.374 * M_earth.to(u.kg)

nb_sw = 6.59e4  *(1/u.cm**3)  
vb_sw = 470000  *(u.m/u.s)
prot_mass = 1.67e-27 * (u.kg)

Omega_pc_E = 0.63
r_exo_E = 6871000  * u.m
rb_IMB = 7647000   * u.m
formfact_o = 1.16
Mag_Earth = 8e22    * (u.A * u.m**2)  

mass = (1.67*10**(-27)) * (u.kg)

magnetic_moments = np.logspace(17, 27, 500)   * (u.A * u.m**2)  
magnetic_m_values_earth_units = magnetic_moments / Mag_Earth


# Converting Constant
earth_atmosphere_in_kg = (1e-6 * M_earth)  
seconds_per_gyr = (1e9 * 365.25 * 24 * 3600)




# Gunell Pickup Escape
 
Q0_pu_H = 5e26   * (1/u.s)
hb_H_B = (k_B * T_exo_H * rb_exo**2) / (G * M_B * m_H)

PU_escape_B = []
pickup_vals_B = []
for M in magnetic_moments:
    R_MP_B = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    if R_MP_B <= rb_exo:
        Q_pu_B = Q0_pu_H
    else:
        num_B = 2*hb_H_B**3 + 2*hb_H_B**2*R_MP_B + hb_H_B*R_MP_B**2
        den_B = 2*hb_H_B**3 + 2*hb_H_B**2*rb_exo + hb_H_B*rb_exo**2
        exp_term_B = np.exp((rb_exo - R_MP_B) / hb_H_B)
        Q_pu_B = Q0_pu_H * (num_B / den_B) * exp_term_B
    pickup_vals_B.append(Q_pu_B)

for Q_pu_B in pickup_vals_B:
    Bmass_loss_rate =  Q_pu_B * mass          # Multiply by mass of hydrogen to get units in kg/s
    Bmass_loss_rate_atmospheres = (Bmass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    PU_escape_B.append(Bmass_loss_rate_atmospheres.value)  
PU_escape_B = u.Quantity(PU_escape_B)



# Cross-Field Ion Loss

Q0_cf_H = 7.7e25   * (1/u.s)

CF_escape_B = []
cf_vals_B = []
for M in magnetic_moments:
    R_MP_B = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    if R_MP_B >= rb_IMB:
        temp_B = 1 - (rb_exo / R_MP_B)
        Omega_pc_B = 4 * np.pi * (1 - np.sqrt(temp_B)) if temp_B >= 0 else 0
    else:
        Omega_pc_B = 0
    Q_cf_B = Q0_cf_H * ((1 - (Omega_pc_B / (4 * np.pi))) / (1 - (Omega_pc_E / (4 * np.pi))))
    cf_vals_B.append(Q_cf_B)

for Q_cf_B in cf_vals_B:
    Bmass_loss_rate =  Q_cf_B * mass          # Multiply by mass of hydrogen to get units in kg/s
    Bmass_loss_rate_atmospheres = (Bmass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    CF_escape_B.append(Bmass_loss_rate_atmospheres.value) 
CF_escape_B = u.Quantity(CF_escape_B)



# Polar Cap Loss

Q0_pc_H = 7.8e25     * (1/u.s)

PC_escape_B = []
pc_vals_B = []
for M in magnetic_moments:
    R_MP_B = (((magnetic_p * formfact_o**2 * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6))
    if R_MP_B > rb_IMB:
        temp_B = 1 - (r_exo_E / R_MP_B)
        Omega_pc_B = 4 * np.pi * (1 - np.sqrt(temp_B)) if temp_B >= 0 else 0
    else:
        Omega_pc_B = 0
    Q_pc_B = Q0_pc_H * (Omega_pc_B / Omega_pc_E) * ((rb_exo / r_exo_E)**2)
    pc_vals_B.append(Q_pc_B)


for Q_pc_B in pc_vals_B:
    Bmass_loss_rate =  Q_pc_B * mass          # Multiply by mass of hydrogen to get units in kg/s
    Bmass_loss_rate_atmospheres = (Bmass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    PC_escape_B.append(Bmass_loss_rate_atmospheres.value)  
PC_escape_B = u.Quantity(PC_escape_B)



# Cusp Escape

Q0_cu_H = 5e24     *( 1/u.s)           
Qmax_cu_H = 5e25   *( 1/u.s) 

# Calculate r_c_E for scaling
nsw = (1e7) * (1 / u.m**3)  
vsw = (604000) * (u.m / u.s)   
r_c_E = ((magnetic_p * Mag_Earth**2) / (8 * np.pi**2 * nsw * vsw**2 * prot_mass))**(1/6)

b_Omega_pc_values = []
CU_escape_B = []
cusp_vals_B = []
for M in magnetic_moments:
    R_MP_B = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    r_c_b = R_MP_B
    
    if rb_IMB <= R_MP_B:
        temp_B = 1 - (r_exo_E / R_MP_B)
        Omega_pc_B = 4 * np.pi * (1 - np.sqrt(temp_B))
    else:
        Omega_pc_B = 0
    b_Omega_pc_values.append(Omega_pc_B)

    
    Q_cu_B = min(Q0_cu_H * (r_c_b / r_c_E)**2, Qmax_cu_H) * (Omega_pc_B / Omega_pc_E) * (rb_exo / r_exo_E)**2
    cusp_vals_B.append(Q_cu_B)

#for Q_cu in cusp_vals:
    Bmass_loss_rate =  Q_cu_B * mass          # Multiply by mass of hydrogen to get units in kg/s
    Bmass_loss_rate_atmospheres = (Bmass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    CU_escape_B.append(Bmass_loss_rate_atmospheres.value)

CU_escape_B = u.Quantity(CU_escape_B)


# Peter Driscoll's Magnetic-Limited Escape 

MPU_escape_B = []
Bmagnetic_limited_vals = []

time = 10  * (u.s)
mass_water = 1e21 * (u.kg)
sigma_coll = (1 * 10**(-17))  * (u.m**2)
energy_eff = (1 / (10.6e44))

for M in magnetic_moments:
    r_mp_B = (((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6))  
    h_exo_b = (k_B * T_exo_H * rb_exo**2) / (G * M_B * mass)
    n_exo_b = (1 / (h_exo_b * sigma_coll))  #.to(1 / u.m**3)
    n_L_b = n_exo_b * np.exp((-rb_exo / h_exo_b) * (1 - (rb_exo / r_mp_B)))
    surface_a_b = (4 * np.pi * r_mp_B**2)
    f_lm_b = (((energy_eff * h_exo_b * surface_a_b * (n_L_b - nb_sw)) / time) * mass_water) #.decompose()
    Bmagnetic_limited_vals.append(f_lm_b)

for f_lm_b in Bmagnetic_limited_vals:
    Bmass_loss_rate =  f_lm_b * mass          # Multiply by mass of hydrogen to get units in kg/s
    Bmass_loss_rate_atmospheres = (Bmass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    MPU_escape_B.append(Bmass_loss_rate_atmospheres.value)

MPU_escape_B =u.Quantity(MPU_escape_B)



# Calculate energy-limited escape rate


a_B = (20.843 * 84e6) * (u.m)     # Used ratio from Eric Agol paper ( a/R*)
M_star = 0.09*M_sun
epsilon = 0.1  # heating efficiency

# Stellar XUV Luminosity
L_bol = 3.828e26 * u.W       # solar luminosity
L_xuv = 3.4e-4 * L_bol       
F_xuv = L_xuv / (4 * np.pi * a_B**2)                      # flux at planet

# Its labeled as the mass density of the plnet. I used the formula to calculate density: mass/V  
rho_B = M_B / ((4/3) * np.pi * R_B**3)                    

# Roche lobe radius (Becker+2020 uses 2 in denominator!)
r_roche_B = a_B * (M_B / (2 * M_star))**(1/3)

# Tidal enhancement factor K
K_B = 1 - (3/2) * (R_B / r_roche_B) + (1/2) * (R_B / r_roche_B)**3

# Energy-limited escape rate (mass loss) in kg/s
mdot_B = epsilon * (R_B / R_B)**2 * (3 * F_xuv) / (4 * G * rho_B * K_B)
mdot_B = mdot_B.to(u.kg / u.s)

seconds_per_gyr = (1e9 * 365.25 * 24 * 3600) * u.s
Bmdot_earth_per_gyr = (mdot_B * seconds_per_gyr / M_earth)


mdot_B_array = u.Quantity(np.full(magnetic_moments.shape, Bmdot_earth_per_gyr.value), unit=Bmdot_earth_per_gyr.unit)

print(mdot_B_array)


# Total Escape  
total_escape_B = PU_escape_B.value + CU_escape_B.value + PC_escape_B.value + CF_escape_B.value

print(total_escape_B)



plt.figure(figsize=(10, 7))

plt.loglog(magnetic_m_values_earth_units, mdot_B_array, label='Energy Limited Escape', color='red', linewidth=3)
plt.loglog(magnetic_m_values_earth_units,  MPU_escape_B, label = 'Driscoll Ion Pickup Escape', color = 'yellow')
plt.loglog(magnetic_m_values_earth_units, PU_escape_B, '--', label='Pickup H Escape', color='orange')
plt.loglog(magnetic_m_values_earth_units, CF_escape_B, '-', label='Cross-field H Escape', color='blue')
plt.loglog(magnetic_m_values_earth_units, PC_escape_B, label='Polar Cap H Escape', color='purple')
plt.loglog(magnetic_m_values_earth_units, CU_escape_B, '--', label='Cusp H Escape', color='hotpink')
plt.loglog(magnetic_m_values_earth_units, total_escape_B, linewidth=2.5, color='black', label='Total H Escape')


plt.xlabel('Magnetic Dipole Moment [A·m²]')
plt.ylabel('Escape Rate [s⁻¹]')
plt.title('TRAPPIST-1b Total Hydrogen Escape vs Magnetic Moment')
plt.legend()
plt.ylim(1e-35, 1e5)
plt.xlim(1e-4, 1e2)

#plt.ylim(1e-5, 1e0)
#plt.xlim(1e2, 1e5)

plt.show()


# In[ ]:





# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import R_earth, M_earth, G, k_B, M_sun
from scipy.interpolate import interp1d

# Constants for TRAPPIST-1c

m_H_C = 3.34e-27 * u.kg
magnetic_p_C = 4 * np.pi * 1e-7 * (u.H / u.m)
mass_C = 1.67e-27 * u.kg

T_exo_H_C = 900 * u.K
R_C_C = 1.097 * R_earth.to(u.m)
rc_exo_C = R_C_C + 396000 * u.m
M_C_C = 1.308 * M_earth.to(u.kg)

nc_sw_C = 2.99e4 * (1 / u.cm**3)
vc_sw_C = 527000 * u.m / u.s
prot_mass_C = 1.67e-27 * u.kg

Omega_pc_E_C = 0.63
r_exo_E_C = 6_871_000 * u.m
rc_IMB_C = 7_647_000 * u.m
formfact_o_C = 1.16
Mag_Earth_C = 8e22 * (u.A * u.m**2)

magnetic_moments_C = np.logspace(17, 27, 500) * (u.A * u.m**2)
magnetic_m_values_earth_units_C = magnetic_moments_C / Mag_Earth_C

# Conversion constants
earth_atmosphere_in_kg_C = 1e-6 * M_earth
seconds_per_gyr_C = 1e9 * 365.25 * 24 * 3600 * u.s



# Pickup Escape



Q0_pu_H_C = 5e26 * (1 / u.s)
hc_H_C = (k_B * T_exo_H_C * rc_exo_C**2) / (G * M_C_C * mass_C)

PU_escape_C = []
pickup_vals_C = []

for M in magnetic_moments_C:
    R_MP = ((magnetic_p_C * M**2) / (8 * np.pi**2 * nc_sw_C * vc_sw_C**2 * prot_mass_C))**(1/6)
    if R_MP <= rc_exo_C:
        Q_pu = Q0_pu_H_C
    else:
        num = 2*hc_H_C**3 + 2*hc_H_C**2*R_MP + hc_H_C*R_MP**2
        den = 2*hc_H_C**3 + 2*hc_H_C**2*rc_exo_C + hc_H_C*rc_exo_C**2
        exp_term = np.exp((rc_exo_C - R_MP) / hc_H_C)
        Q_pu = Q0_pu_H_C * (num / den) * exp_term
    pickup_vals_C.append(Q_pu)

for Q_pu in pickup_vals_C:
    mass_loss_rate = Q_pu * mass_C
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg_C) * seconds_per_gyr_C
    PU_escape_C.append(mass_loss_rate_atmospheres.value)

PU_escape_C = u.Quantity(PU_escape_C)



# Cross-Field Ion Loss



Q0_cf_H_C = 7.7e25 * (1 / u.s)

CF_escape_C = []
cf_vals_C = []

for M in magnetic_moments_C:
    R_MP = ((magnetic_p_C * M**2) / (8 * np.pi**2 * nc_sw_C * vc_sw_C**2 * prot_mass_C))**(1/6)
    if R_MP >= rc_IMB_C:
        temp = 1 - (rc_exo_C / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cf = Q0_cf_H_C * ((1 - (Omega_pc / (4*np.pi))) / (1 - (Omega_pc_E_C / (4*np.pi))))
    cf_vals_C.append(Q_cf)

for Q_cf in cf_vals_C:
    mass_loss_rate = Q_cf * mass_C
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg_C) * seconds_per_gyr_C
    CF_escape_C.append(mass_loss_rate_atmospheres.value)

CF_escape_C = u.Quantity(CF_escape_C)



# Polar Cap Loss



Q0_pc_H_C = 7.8e25 * (1 / u.s)

PC_escape_C = []
pc_vals_C = []

for M in magnetic_moments_C:
    R_MP = ((magnetic_p_C * formfact_o_C**2 * M**2) / (8 * np.pi**2 * nc_sw_C * vc_sw_C**2 * prot_mass_C))**(1/6)
    if R_MP > rc_IMB_C:
        temp = 1 - (r_exo_E_C / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_pc = Q0_pc_H_C * (Omega_pc / Omega_pc_E_C) * ((rc_exo_C / r_exo_E_C)**2)
    pc_vals_C.append(Q_pc)

for Q_pc in pc_vals_C:
    mass_loss_rate = Q_pc * mass_C
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg_C) * seconds_per_gyr_C
    PC_escape_C.append(mass_loss_rate_atmospheres.value)

PC_escape_C = u.Quantity(PC_escape_C)



# Cusp Escape



Q0_cu_H_C = 5e24 * (1 / u.s)
Qmax_cu_H_C = 5e25 * (1 / u.s)

nsw_C = 1e7 * (1 / u.m**3)
vsw_C = 604000 * u.m / u.s
r_c_E_C = ((magnetic_p_C * Mag_Earth_C**2) / (8*np.pi**2 * nsw_C * vsw_C**2 * prot_mass_C))**(1/6)

CU_escape_C = []
cusp_vals_C = []

for M in magnetic_moments_C:
    R_MP = ((magnetic_p_C * M**2) / (8*np.pi**2 * nc_sw_C * vc_sw_C**2 * prot_mass_C))**(1/6)
    r_c_c = R_MP
    if R_MP > rc_IMB_C:
        temp = 1 - (r_exo_E_C / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cu = min(Q0_cu_H_C * (r_c_c / r_c_E_C)**2, Qmax_cu_H_C) * (Omega_pc / Omega_pc_E_C) * (rc_exo_C / r_exo_E_C)**2
    cusp_vals_C.append(Q_cu)

for Q_cu in cusp_vals_C:
    mass_loss_rate = Q_cu * mass_C
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg_C) * seconds_per_gyr_C
    CU_escape_C.append(mass_loss_rate_atmospheres.value)

CU_escape_C = u.Quantity(CU_escape_C)


# Driscoll Ion Pickup Escape



MPU_escape_C = []
magnetic_limited_vals_C = []

time_C = 10 * u.s
mass_water_C = 1e21 * u.kg
sigma_coll_C = 1e-17 * u.m**2
energy_eff_C = 1 / 10.6e44

for M in magnetic_moments_C:
    r_mp = ((magnetic_p_C * M**2) / (8*np.pi**2 * nc_sw_C * vc_sw_C**2 * prot_mass_C))**(1/6)
    h_exo_c = (k_B * T_exo_H_C * rc_exo_C**2) / (G * M_C_C * mass_C)
    n_exo_c = (1 / (h_exo_c * sigma_coll_C))
    n_L_c = n_exo_c * np.exp((-rc_exo_C / h_exo_c) * (1 - (rc_exo_C / r_mp)))
    surface_a_c = 4 * np.pi * r_mp**2
    f_lm_c = ((energy_eff_C * h_exo_c * surface_a_c * (n_L_c - nc_sw_C) / time_C) * mass_water_C)
    magnetic_limited_vals_C.append(f_lm_c)

for f_lm_c in magnetic_limited_vals_C:
    mass_loss_rate = f_lm_c * mass_C
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg_C) * seconds_per_gyr_C
    MPU_escape_C.append(mass_loss_rate_atmospheres.value)

MPU_escape_C = u.Quantity(MPU_escape_C)



# Energy-Limited Escape



a_C = 28.549 * 84e6 * u.m
M_star_C = 0.09 * M_sun
epsilon_C = 0.1

L_bol_C = 3.828e26 * u.W
L_xuv_C = 3.4e-4 * L_bol_C
F_xuv_C = L_xuv_C / (4 * np.pi * a_C**2)

rho_C = M_C_C / ((4/3) * np.pi * R_C_C**3)
r_roche_C = a_C * (M_C_C / (2 * M_star_C))**(1/3)
K_C = 1 - (3/2)*(R_C_C / r_roche_C) + (1/2)*(R_C_C / r_roche_C)**3

mdot_C = epsilon_C * (R_C_C / R_C_C)**2 * (3 * F_xuv_C) / (4 * G * rho_C * K_C)
mdot_C = mdot_C.to(u.kg / u.s)
mdot_earth_per_gyr_C = (mdot_C * seconds_per_gyr_C / M_earth)
mdot_array_C = u.Quantity(np.full(magnetic_moments_C.shape, mdot_earth_per_gyr_C.value), unit=mdot_earth_per_gyr_C.unit)



# Total Escape



total_escape_C = PU_escape_C.value + CU_escape_C.value + PC_escape_C.value + CF_escape_C.value


# Plot


plt.figure(figsize=(10, 7))

plt.loglog(magnetic_m_values_earth_units_C, mdot_array_C, label='Energy Limited Escape', color='red', linewidth=3)
plt.loglog(magnetic_m_values_earth_units_C, MPU_escape_C, label='Driscoll Ion Pickup Escape', color='yellow')
plt.loglog(magnetic_m_values_earth_units_C, PU_escape_C, '--', label='Pickup H Escape', color='orange')
plt.loglog(magnetic_m_values_earth_units_C, CF_escape_C, '-', label='Cross-field H Escape', color='blue')
plt.loglog(magnetic_m_values_earth_units_C, PC_escape_C, label='Polar Cap H Escape', color='purple')
plt.loglog(magnetic_m_values_earth_units_C, CU_escape_C, '--', label='Cusp H Escape', color='hotpink')
plt.loglog(magnetic_m_values_earth_units_C, total_escape_C, linewidth=2.5, color='black', label='Total H Escape')

plt.xlabel('Magnetic Dipole Moment [A·m²]', fontsize=14)
plt.ylabel('Escape Rate [Earth Atmospheres / Gyr]', fontsize=14)
plt.title('TRAPPIST-1c Total Hydrogen Escape vs Magnetic Moment', fontsize=16)
plt.legend(fontsize=12)
plt.ylim(1e-35, 1e5)
plt.xlim(1e-4, 1e2)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import R_earth, M_earth, G, k_B, M_sun
from scipy.interpolate import interp1d

# Constants for TRAPPIST-1d

m_H_D = 3.34e-27 * u.kg
magnetic_p_D = 4 * np.pi * 1e-7 * (u.H / u.m)

T_exo_H_D = 900 * u.K
R_D_D = 0.788 * R_earth.to(u.m)
rd_exo_D = R_D_D + 396000 * u.m
M_D_D = 0.388 * M_earth.to(u.kg)

nd_sw_D = 1.20e4 * (1 / u.cm**3)
vd_sw_D = 566000 * u.m / u.s
prot_mass_D = 1.67e-27 * u.kg

Omega_pc_E_D = 0.63
r_exo_E_D = 6_871_000 * u.m
rd_IMB_D = 7_647_000 * u.m
formfact_o_D = 1.16
Mag_Earth_D = 8e22 * u.A * u.m**2

magnetic_moments_D = np.logspace(17, 27, 500) * u.A * u.m**2
magnetic_m_values_earth_units_D = magnetic_moments_D / Mag_Earth_D

# Conversion constants
earth_atmosphere_in_kg_D = 1e-6 * M_earth
seconds_per_gyr_D = 1e9 * 365.25 * 24 * 3600 * u.s



# Pickup Escape



Q0_pu_H_D = 5e26 * (1 / u.s)
hd_H_D = (k_B * T_exo_H_D * rd_exo_D**2) / (G * M_D_D * m_H_D)

PU_escape_D = []
pickup_vals_D = []

for M in magnetic_moments_D:
    R_MP = ((magnetic_p_D * M**2) / (8 * np.pi**2 * nd_sw_D * vd_sw_D**2 * prot_mass_D))**(1/6)
    if R_MP <= rd_exo_D:
        Q_pu = Q0_pu_H_D
    else:
        num = 2*hd_H_D**3 + 2*hd_H_D**2*R_MP + hd_H_D*R_MP**2
        den = 2*hd_H_D**3 + 2*hd_H_D**2*rd_exo_D + hd_H_D*rd_exo_D**2
        exp_term = np.exp((rd_exo_D - R_MP) / hd_H_D)
        Q_pu = Q0_pu_H_D * (num / den) * exp_term
    pickup_vals_D.append(Q_pu)

for Q_pu in pickup_vals_D:
    mass_loss_rate = Q_pu * m_H_D
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg_D) * seconds_per_gyr_D
    PU_escape_D.append(mass_loss_rate_atmospheres.value)

PU_escape_D = u.Quantity(PU_escape_D)



# Cross-Field Ion Loss



Q0_cf_H_D = 7.7e25 * (1 / u.s)

CF_escape_D = []
cf_vals_D = []

for M in magnetic_moments_D:
    R_MP = ((magnetic_p_D * M**2) / (8 * np.pi**2 * nd_sw_D * vd_sw_D**2 * prot_mass_D))**(1/6)
    if R_MP >= rd_IMB_D:
        temp = 1 - (rd_exo_D / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cf = Q0_cf_H_D * ((1 - (Omega_pc / (4*np.pi))) / (1 - (Omega_pc_E_D / (4*np.pi))))
    cf_vals_D.append(Q_cf)

for Q_cf in cf_vals_D:
    mass_loss_rate = Q_cf * m_H_D
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg_D) * seconds_per_gyr_D
    CF_escape_D.append(mass_loss_rate_atmospheres.value)

CF_escape_D = u.Quantity(CF_escape_D)



# Polar Cap Loss



Q0_pc_H_D = 7.8e25 * (1 / u.s)

PC_escape_D = []
pc_vals_D = []

for M in magnetic_moments_D:
    R_MP = ((magnetic_p_D * formfact_o_D**2 * M**2) / (8 * np.pi**2 * nd_sw_D * vd_sw_D**2 * prot_mass_D))**(1/6)
    if R_MP > rd_IMB_D:
        temp = 1 - (r_exo_E_D / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_pc = Q0_pc_H_D * (Omega_pc / Omega_pc_E_D) * ((rd_exo_D / r_exo_E_D)**2)
    pc_vals_D.append(Q_pc)

for Q_pc in pc_vals_D:
    mass_loss_rate = Q_pc * m_H_D
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg_D) * seconds_per_gyr_D
    PC_escape_D.append(mass_loss_rate_atmospheres.value)

PC_escape_D = u.Quantity(PC_escape_D)



# Cusp Escape



Q0_cu_H_D = 5e24 * (1 / u.s)
Qmax_cu_H_D = 5e25 * (1 / u.s)

nsw_D = 1e7 * (1 / u.m**3)
vsw_D = 604000 * u.m / u.s
r_c_E_D = ((magnetic_p_D * Mag_Earth_D**2) / (8*np.pi**2 * nsw_D * vsw_D**2 * prot_mass_D))**(1/6)

CU_escape_D = []
cusp_vals_D = []

for M in magnetic_moments_D:
    R_MP = ((magnetic_p_D * M**2) / (8*np.pi**2 * nd_sw_D * vd_sw_D**2 * prot_mass_D))**(1/6)
    r_c_d = R_MP
    if R_MP > rd_IMB_D:
        temp = 1 - (r_exo_E_D / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cu = min(Q0_cu_H_D * (r_c_d / r_c_E_D)**2, Qmax_cu_H_D) * (Omega_pc / Omega_pc_E_D) * (rd_exo_D / r_exo_E_D)**2
    cusp_vals_D.append(Q_cu)

for Q_cu in cusp_vals_D:
    mass_loss_rate = Q_cu * m_H_D
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg_D) * seconds_per_gyr_D
    CU_escape_D.append(mass_loss_rate_atmospheres.value)

CU_escape_D = u.Quantity(CU_escape_D)



# Driscoll Ion Pickup Escape



MPU_escape_D = []
magnetic_limited_vals_D = []

time_D = 10 * u.s
mass_water_D = 1e21 * u.kg
sigma_coll_D = 1e-17 * u.m**2
energy_eff_D = 1 / 10.6e44

for M in magnetic_moments_D:
    r_mp = ((magnetic_p_D * M**2) / (8*np.pi**2 * nd_sw_D * vd_sw_D**2 * prot_mass_D))**(1/6)
    h_exo_d = (k_B * T_exo_H_D * rd_exo_D**2) / (G * M_D_D * m_H_D)
    n_exo_d = 1 / (h_exo_d * sigma_coll_D)
    n_L_d = n_exo_d * np.exp((-rd_exo_D / h_exo_d) * (1 - (rd_exo_D / r_mp)))
    surface_a_d = 4 * np.pi * r_mp**2
    f_lm_d = ((energy_eff_D * h_exo_d * surface_a_d * (n_L_d - nd_sw_D) / time_D) * mass_water_D)
    magnetic_limited_vals_D.append(f_lm_d)

for f_lm_d in magnetic_limited_vals_D:
    mass_loss_rate = f_lm_d * m_H_D
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg_D) * seconds_per_gyr_D
    MPU_escape_D.append(mass_loss_rate_atmospheres.value)

MPU_escape_D = u.Quantity(MPU_escape_D)



# Energy-Limited Escape



a_D = 40.216 * 84e6 * u.m
M_star_D = 0.09 * M_sun
epsilon_D = 0.1

L_bol_D = 3.828e26 * u.W
L_xuv_D = 3.4e-4 * L_bol_D
F_xuv_D = L_xuv_D / (4 * np.pi * a_D**2)

rho_D = M_D_D / ((4/3) * np.pi * R_D_D**3)
r_roche_D = a_D * (M_D_D / (2 * M_star_D))**(1/3)
K_D = 1 - (3/2)*(R_D_D / r_roche_D) + (1/2)*(R_D_D / r_roche_D)**3

mdot_D = epsilon_D * (R_D_D / R_D_D)**2 * (3 * F_xuv_D) / (4 * G * rho_D * K_D)
mdot_D = mdot_D.to(u.kg / u.s)
mdot_earth_per_gyr_D = (mdot_D * seconds_per_gyr_D / M_earth)
mdot_array_D = u.Quantity(np.full(magnetic_moments_D.shape, mdot_earth_per_gyr_D.value), unit=mdot_earth_per_gyr_D.unit)



# Total Escape



total_escape_D = PU_escape_D.value + CU_escape_D.value + PC_escape_D.value + CF_escape_D.value


# Plot


plt.figure(figsize=(10, 7))

plt.loglog(magnetic_m_values_earth_units_D, mdot_array_D, label='Energy Limited Escape', color='red', linewidth=3)
plt.loglog(magnetic_m_values_earth_units_D, MPU_escape_D, label='Driscoll Ion Pickup Escape', color='yellow')
plt.loglog(magnetic_m_values_earth_units_D, PU_escape_D, '--', label='Pickup H Escape', color='orange')
plt.loglog(magnetic_m_values_earth_units_D, CF_escape_D, '-', label='Cross-field H Escape', color='blue')
plt.loglog(magnetic_m_values_earth_units_D, PC_escape_D, label='Polar Cap H Escape', color='purple')
plt.loglog(magnetic_m_values_earth_units_D, CU_escape_D, '--', label='Cusp H Escape', color='hotpink')
plt.loglog(magnetic_m_values_earth_units_D, total_escape_D, linewidth=2.5, color='black', label='Total H Escape')

plt.xlabel('Magnetic Dipole Moment [A·m²]', fontsize=14)
plt.ylabel('Escape Rate [Earth Atmospheres / Gyr]', fontsize=14)
plt.title('TRAPPIST-1d Total Hydrogen Escape vs Magnetic Moment', fontsize=16)
plt.legend(fontsize=12)
plt.ylim(1e-35, 1e5)
plt.xlim(1e-4, 1e2)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import R_earth, M_earth, G, k_B, M_sun
from scipy.interpolate import interp1d


m_H_TE = (3.34 * 10**(-27)) * (u.kg)
magnetic_p_TE = 4 * np.pi * 1e-7  * (u.H / u.m)

T_exo_H_TE = 900 * u.K
R_EE_TE = 0.920 * R_earth.to(u.m)
ree_exo_TE = R_EE_TE + (396000 * u.m)
M_EE_TE = 0.692 * M_earth.to(u.kg)

ne_sw_TE = 5.79e3 * (1/u.cm**3)
ve_sw_TE = 604000 * u.m/u.s
prot_mass_TE = 1.67e-27 * u.kg

Omega_pc_E_TE = 0.63
r_exo_E_TE = 6871000 * u.m
re_IMB_TE = 7647000 * u.m
formfact_o_TE = 1.16
Mag_Earth_TE = 8e22 * (u.A * u.m**2)
mass_TE = (1.67*10**(-27)) * (u.kg)

magnetic_moments_TE = np.logspace(17, 27, 500) * (u.A * u.m**2)
magnetic_m_values_earth_units_TE = magnetic_moments_TE / Mag_Earth_TE

# Conversions
earth_atmosphere_in_kg_TE = (1e-6 * M_earth)
seconds_per_gyr_TE = (1e9 * 365.25 * 24 * 3600)



# Pickup Escape



Q0_pu_H_TE = 5e26 * (1/u.s)
he_H_TE = (k_B * T_exo_H_TE * ree_exo_TE**2) / (G * M_EE_TE * m_H_TE)

PU_escape_TE = []
for M in magnetic_moments_TE:
    R_MP = ((magnetic_p_TE * M**2) / (8 * np.pi**2 * ne_sw_TE * ve_sw_TE**2 * prot_mass_TE))**(1/6)
    if R_MP <= ree_exo_TE:
        Q_pu = Q0_pu_H_TE
    else:
        num = 2*he_H_TE**3 + 2*he_H_TE**2*R_MP + he_H_TE*R_MP**2
        den = 2*he_H_TE**3 + 2*he_H_TE**2*ree_exo_TE + he_H_TE*ree_exo_TE**2
        exp_term = np.exp((ree_exo_TE - R_MP) / he_H_TE)
        Q_pu = Q0_pu_H_TE * (num / den) * exp_term
    mass_loss_rate = Q_pu * mass_TE
    atmos_loss = (mass_loss_rate / earth_atmosphere_in_kg_TE) * seconds_per_gyr_TE
    PU_escape_TE.append(atmos_loss.value)
PU_escape_TE = u.Quantity(PU_escape_TE)



# Cross-Field Ion Loss


Q0_cf_H_TE = 7.7e25 * (1/u.s)

CF_escape_TE = []
for M in magnetic_moments_TE:
    R_MP = ((magnetic_p_TE * M**2) / (8 * np.pi**2 * ne_sw_TE * ve_sw_TE**2 * prot_mass_TE))**(1/6)
    if R_MP >= re_IMB_TE:
        temp = 1 - (ree_exo_TE / R_MP)
        Omega_pc = 4*np.pi*(1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cf = Q0_cf_H_TE * ((1 - (Omega_pc / (4*np.pi))) / (1 - (Omega_pc_E_TE / (4*np.pi))))
    mass_loss_rate = Q_cf * mass_TE
    atmos_loss = (mass_loss_rate / earth_atmosphere_in_kg_TE) * seconds_per_gyr_TE
    CF_escape_TE.append(atmos_loss.value)
CF_escape_TE = u.Quantity(CF_escape_TE)



# Polar Cap Escape


Q0_pc_H_TE = 7.8e25 * (1/u.s)

PC_escape_TE = []
for M in magnetic_moments_TE:
    R_MP = ((magnetic_p_TE * formfact_o_TE**2 * M**2) / (8 * np.pi**2 * ne_sw_TE * ve_sw_TE**2 * prot_mass_TE))**(1/6)
    if R_MP > re_IMB_TE:
        temp = 1 - (r_exo_E_TE / R_MP)
        Omega_pc = 4*np.pi*(1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_pc = Q0_pc_H_TE * (Omega_pc / Omega_pc_E_TE) * ((ree_exo_TE / r_exo_E_TE)**2)
    mass_loss_rate = Q_pc * mass_TE
    atmos_loss = (mass_loss_rate / earth_atmosphere_in_kg_TE) * seconds_per_gyr_TE
    PC_escape_TE.append(atmos_loss.value)
PC_escape_TE = u.Quantity(PC_escape_TE)



# Cusp Escape


Q0_cu_H_TE = 5e24 * (1/u.s)
Qmax_cu_H_TE = 5e25 * (1/u.s)

nsw_E = (1e7) * (1 / u.m**3)  
vsw_E = (604000) * (u.m / u.s)   
r_c_E = ((magnetic_p_TE * Mag_Earth_TE**2) / (8 * np.pi**2 * nsw_E * vsw_E**2 * prot_mass_TE))**(1/6)

CU_escape_TE = []
cusp_vals_TE = []

for M in magnetic_moments_TE:
    R_MP_TE = ((magnetic_p_TE * M**2) / (8 * np.pi**2 * ne_sw_TE * ve_sw_TE**2 * prot_mass_TE))**(1/6)
    r_c_E_TE = R_MP_TE
    if R_MP_TE > re_IMB_TE:
        temp_TE = 1 - (r_exo_E_TE / R_MP_TE)
        Omega_pc_TE = 4*np.pi*(1 - np.sqrt(temp_TE)) if temp_TE >= 0 else 0
    else:
        Omega_pc_TE = 0
    Q_cu_TE = min(Q0_cu_H_TE * (r_c_E_TE / r_c_E)**2, Qmax_cu_H_TE) * (Omega_pc_TE / Omega_pc_E_TE) * (ree_exo_TE / r_exo_E_TE)**2
    cusp_vals_TE.append(Q_cu_TE)
    
for Q_cu_TE in cusp_vals_TE:    
    mass_loss_rate_TE = Q_cu_TE * mass_TE
    atmos_loss_TE = (mass_loss_rate_TE / earth_atmosphere_in_kg_TE) * seconds_per_gyr_TE
    CU_escape_TE.append(atmos_loss_TE.value)
CU_escape_TE = u.Quantity(CU_escape_TE)



# Driscoll Ion Pickup Escape


MPU_escape_TE = []
time_TE = 10 * u.s
mass_water_TE = 1e21 * u.kg
sigma_coll_TE = (1e-17) * u.m**2
energy_eff_TE = 1 / (10.6e44)

for M in magnetic_moments_TE:
    r_mp = ((magnetic_p_TE * M**2) / (8 * np.pi**2 * ne_sw_TE * ve_sw_TE**2 * prot_mass_TE))**(1/6)
    h_exo = (k_B * T_exo_H_TE * ree_exo_TE**2) / (G * M_EE_TE * m_H_TE)
    n_exo = 1 / (h_exo * sigma_coll_TE)
    n_L = n_exo * np.exp((-ree_exo_TE / h_exo) * (1 - (ree_exo_TE / r_mp)))
    surface_a = 4 * np.pi * r_mp**2
    f_lm = ((energy_eff_TE * h_exo * surface_a * (n_L - ne_sw_TE)) / time_TE) * mass_water_TE
    mass_loss_rate = f_lm * mass_TE
    atmos_loss = (mass_loss_rate / earth_atmosphere_in_kg_TE) * seconds_per_gyr_TE
    MPU_escape_TE.append(atmos_loss.value)
MPU_escape_TE = u.Quantity(MPU_escape_TE)


# Energy-Limited Escape


a_TE = (52.855 * 84e6) * u.m
M_star_TE = 0.09 * M_sun
epsilon_TE = 0.1

L_bol_TE = 3.828e26 * u.W
L_xuv_TE = 3.4e-4 * L_bol_TE
F_xuv_TE = L_xuv_TE / (4 * np.pi * a_TE**2)

rho_TE = M_EE_TE / ((4/3) * np.pi * R_EE_TE**3)
r_roche_TE = a_TE * (M_EE_TE / (2 * M_star_TE))**(1/3)
K_TE = 1 - (3/2)*(R_EE_TE/r_roche_TE) + (1/2)*(R_EE_TE/r_roche_TE)**3

mdot_TE = epsilon_TE * (3*F_xuv_TE) / (4 * G * rho_TE * K_TE)
mdot_TE = mdot_TE.to(u.kg/u.s)

seconds_per_gyr_TE = (1e9 * 365.25 * 24 * 3600) * u.s
mdot_earth_per_gyr_TE = (mdot_TE * seconds_per_gyr_TE / M_earth)
mdot_array_TE = u.Quantity(np.full(magnetic_moments_TE.shape, mdot_earth_per_gyr_TE.value),
                           unit=mdot_earth_per_gyr_TE.unit)



# Total Escape


total_escape_TE = (PU_escape_TE.value + CU_escape_TE.value +
                   PC_escape_TE.value + CF_escape_TE.value)


# Plot

plt.figure(figsize=(10, 7))
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

plt.loglog(magnetic_m_values_earth_units_TE, mdot_array_TE, label='Energy-Limited Escape', color='red', linewidth=3)
plt.loglog(magnetic_m_values_earth_units_TE, MPU_escape_TE, label='Driscoll Ion Pickup', color='gold')
plt.loglog(magnetic_m_values_earth_units_TE, PU_escape_TE, '--', label='Pickup H Escape', color='orange')
plt.loglog(magnetic_m_values_earth_units_TE, CF_escape_TE, '-', label='Cross-field H Escape', color='blue')
plt.loglog(magnetic_m_values_earth_units_TE, PC_escape_TE, label='Polar Cap H Escape', color='purple')
plt.loglog(magnetic_m_values_earth_units_TE, CU_escape_TE, '--', label='Cusp H Escape', color='hotpink')
plt.loglog(magnetic_m_values_earth_units_TE, total_escape_TE, linewidth=2.5, color='black', label='Total H Escape')

plt.xlabel('Magnetic Dipole Moment [Earth Units]', fontsize=16)
plt.ylabel('Escape Rate [Earth Atmospheres / Gyr]', fontsize=16)
plt.title('TRAPPIST-1e Total Hydrogen Escape vs Magnetic Moment', fontsize=18, pad=15)
plt.legend(fontsize=12, loc='best', frameon=True)
plt.ylim(1e-35, 1e5)
plt.xlim(1e-4, 1e2)
plt.tight_layout()
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import R_earth, M_earth, G, k_B, M_sun
from scipy.interpolate import interp1d


m_H_F = (3.34 * 10**(-27)) * (u.kg)
magnetic_p_F = 4 * np.pi * 1e-7  * (u.H / u.m) 

T_exo_H_F = 900 * u.K
R_F = 1.045 * R_earth.to(u.m)
M_F = 1.039 * M_earth.to(u.kg)

nf_sw_F = 2.99e3 *(1/u.cm**3)      
vf_sw_F = 624000  * u.m/u.s        
prot_mass_F = 1.67e-27 *u.kg

Omega_pc_E_F = 0.63
r_exo_E_F = 6871000 *u.m
rf_IMB_F = 7647000  *u.m           
formfact_o_F = 1.16
Mag_Earth_F = 8e22    * (u.A * u.m**2)  
mass_h_F = (3.34 * 10**(-27)) * u.kg
mass_F = (1.67*10**(-27)) * (u.kg)
z_exo_F = 396 * (u.km)
rf_exo_F = R_F + z_exo_F 

magnetic_moments_F = np.logspace(17, 27, 500)   * (u.A * u.m**2)  
magnetic_m_values_earth_units_F = magnetic_moments_F / Mag_Earth_F

# Converting Constant
earth_atmosphere_in_kg_F = (1e-6 * M_earth)  
seconds_per_gyr_F = (1e9 * 365.25 * 24 * 3600)


# Driscoll Ion Pickup Escape


MPU_escape_F = []
magnetic_limited_vals_F = []

time_F = 10  * (u.s)
mass_water_F = 1e21 * (u.kg)
sigma_coll_F = (1 * 10**(-17))  * (u.m**2)
energy_eff_F = (1 / (10.6e44))

for M_Fm in magnetic_moments_F:
    r_mp_F = (((magnetic_p_F * M_Fm**2) / (8 * np.pi**2 * nf_sw_F * vf_sw_F**2 * prot_mass_F))**(1/6))  
    h_exo_f_F = (k_B * T_exo_H_F * rf_exo_F**2) / (G * M_F * mass_h_F)
    n_exo_f_F = (1 / (h_exo_f_F * sigma_coll_F))
    n_L_f_F = n_exo_f_F * np.exp((-rf_exo_F / h_exo_f_F) * (1 - (rf_exo_F / r_mp_F)))
    surface_a_f_F = (4 * np.pi * r_mp_F**2)
    f_lm_f_F = (((energy_eff_F * h_exo_f_F * surface_a_f_F * (n_L_f_F - nf_sw_F)) / time_F) * mass_water_F)
    magnetic_limited_vals_F.append(f_lm_f_F)

for f_lm_f_F in magnetic_limited_vals_F:
    mass_loss_rate_F =  f_lm_f_F * mass_F          
    mass_loss_rate_atmospheres_F = (mass_loss_rate_F / earth_atmosphere_in_kg_F) * seconds_per_gyr_F  
    MPU_escape_F.append(mass_loss_rate_atmospheres_F.value)

MPU_escape_F =u.Quantity(MPU_escape_F)



# Gunell Pickup Escape


Q0_pu_H_F = 5e26 *( 1/u.s) 
hf_H_F = (k_B * T_exo_H_F * rf_exo_F**2) / (G * M_F * m_H_F)

PU_escape_F = []
pickup_vals_F = []
for M_Fm in magnetic_moments_F:
    R_MP_F = ((magnetic_p_F * M_Fm**2) / (8 * np.pi**2 * nf_sw_F * vf_sw_F**2 * prot_mass_F))**(1/6)
    if R_MP_F <= rf_exo_F:
        Q_pu_F = Q0_pu_H_F
    else:
        num_F = 2*hf_H_F**3 + 2*hf_H_F**2*R_MP_F + hf_H_F*R_MP_F**2
        den_F = 2*hf_H_F**3 + 2*hf_H_F**2*rf_exo_F + hf_H_F*rf_exo_F**2
        exp_term_F = np.exp((rf_exo_F - R_MP_F) / hf_H_F)
        Q_pu_F = (Q0_pu_H_F * (num_F / den_F) * exp_term_F)
    pickup_vals_F.append(Q_pu_F)

for Q_pu_F in pickup_vals_F:
    mass_loss_rate_F =  Q_pu_F * mass_F          
    mass_loss_rate_atmospheres_F = (mass_loss_rate_F / earth_atmosphere_in_kg_F) * seconds_per_gyr_F  
    PU_escape_F.append(mass_loss_rate_atmospheres_F.value)  

PU_escape_F = u.Quantity(PU_escape_F)



# Cross-Field Ion Loss


Q0_cf_H_F = 7.7e25   *( 1/u.s)     

CF_escape_F = []
cf_vals_F = []
for M_Fm in magnetic_moments_F:
    R_MP_F = ((magnetic_p_F * M_Fm**2) / (8 * np.pi**2 * nf_sw_F * vf_sw_F**2 * prot_mass_F))**(1/6)
    if R_MP_F >= rf_IMB_F:
        temp_F = 1 - (rf_exo_F / R_MP_F)
        Omega_pc_F = 4 * np.pi * (1 - np.sqrt(temp_F)) if temp_F >= 0 else 0
    else:
        Omega_pc_F = 0
    Q_cf_F = Q0_cf_H_F * ((1 - (Omega_pc_F / (4 * np.pi))) / (1 - (Omega_pc_E_F / (4 * np.pi))))
    cf_vals_F.append(Q_cf_F)

for Q_cf_F in cf_vals_F:
    mass_loss_rate_F =  Q_cf_F * mass_F          
    mass_loss_rate_atmospheres_F = (mass_loss_rate_F / earth_atmosphere_in_kg_F) * seconds_per_gyr_F  
    CF_escape_F.append(mass_loss_rate_atmospheres_F.value) 

CF_escape_F = u.Quantity(CF_escape_F)


# Polar Cap Loss


Q0_pc_H_F = 7.8e25  *( 1/u.s)       

PC_escape_F = []
pc_vals_F = []
for M_Fm in magnetic_moments_F:
    R_MP_F = (((magnetic_p_F * formfact_o_F**2 * M_Fm**2) / (8 * np.pi**2 * nf_sw_F * vf_sw_F**2 * prot_mass_F))**(1/6))
    if R_MP_F > rf_IMB_F:
        temp_F = 1 - (r_exo_E_F / R_MP_F)
        Omega_pc_F = 4 * np.pi * (1 - np.sqrt(temp_F)) if temp_F >= 0 else 0
    else:
        Omega_pc_F = 0
    Q_pc_F = Q0_pc_H_F * (Omega_pc_F / Omega_pc_E_F) * ((rf_exo_F / r_exo_E_F)**2)
    pc_vals_F.append(Q_pc_F)

for Q_pc_F in pc_vals_F:
    mass_loss_rate_F =  Q_pc_F * mass_F          
    mass_loss_rate_atmospheres_F = (mass_loss_rate_F / earth_atmosphere_in_kg_F) * seconds_per_gyr_F  
    PC_escape_F.append(mass_loss_rate_atmospheres_F.value)  

PC_escape_F = u.Quantity(PC_escape_F)



# Cusp Escape


Q0_cu_H_F = 5e24     *( 1/u.s)           
Qmax_cu_H_F = 5e25   *( 1/u.s)         

nsw_F = (1e7) * (1 / u.m**3)  
vsw_F = (604000) * (u.m / u.s)   
r_c_E_F = ((magnetic_p_F * Mag_Earth_F**2) / (8 * np.pi**2 * nsw_F * vsw_F**2 * prot_mass_F))**(1/6)

CU_escape_F = []
cusp_vals_F = []
for M_Fm in magnetic_moments_F:
    R_MP_F = ((magnetic_p_F * M_Fm**2) / (8 * np.pi**2 * nf_sw_F * vf_sw_F**2 * prot_mass_F))**(1/6)
    r_c_f_F = R_MP_F
    if R_MP_F > rf_IMB_F:
        temp_F = 1 - (r_exo_E_F / R_MP_F)
        Omega_pc_F = 4 * np.pi * (1 - np.sqrt(temp_F)) if temp_F >= 0 else 0
    else:
        Omega_pc_F = 0
    Q_cu_F = min(Q0_cu_H_F * (r_c_f_F / r_c_E_F)**2, Qmax_cu_H_F) * (Omega_pc_F / Omega_pc_E_F) * (rf_exo_F / r_exo_E_F)**2
    cusp_vals_F.append(Q_cu_F)

for Q_cu_F in cusp_vals_F:
    mass_loss_rate_F =  Q_cu_F * mass_F          
    mass_loss_rate_atmospheres_F = (mass_loss_rate_F / earth_atmosphere_in_kg_F) * seconds_per_gyr_F  
    CU_escape_F.append(mass_loss_rate_atmospheres_F.value)

CU_escape_F = u.Quantity(CU_escape_F)



# Energy-limited escape


a_F = (69.543 * 84e6) * (u.m)     
M_star_F = 0.09*M_sun
epsilon_F = 0.1  

L_bol_F = 3.828e26 * u.W       
L_xuv_F = 3.4e-4 * L_bol_F       
F_xuv_F = L_xuv_F / (4 * np.pi * a_F**2)                      

rho_F = M_F / ((4/3) * np.pi * R_F**3)                    
r_roche_F = a_F * (M_F / (2 * M_star_F))**(1/3)

K_F = 1 - (3/2) * (R_F / r_roche_F) + (1/2) * (R_F / r_roche_F)**3

mdot_F = epsilon_F * (R_F / R_F)**2 * (3 * F_xuv_F) / (4 * G * rho_F * K_F)
mdot_F = mdot_F.to(u.kg / u.s)

seconds_per_gyr_F = (1e9 * 365.25 * 24 * 3600) * u.s
mdot_earth_per_gyr_F = (mdot_F * seconds_per_gyr_F / M_earth)

mdot_array_F = u.Quantity(np.full(magnetic_moments_F.shape, mdot_earth_per_gyr_F.value), unit=mdot_earth_per_gyr_F.unit)


# Total Escape  


total_escape_F = PU_escape_F.value + CU_escape_F.value + PC_escape_F.value + CF_escape_F.value 


# Plot Section


plt.figure(figsize=(10, 7))

plt.loglog(magnetic_m_values_earth_units_F, mdot_array_F, color='red', linewidth=3, label='Energy-Limited Escape')
plt.loglog(magnetic_m_values_earth_units_F, MPU_escape_F, label = 'Driscoll Ion Pickup Escape', color = 'yellow')
plt.loglog(magnetic_m_values_earth_units_F, PU_escape_F, '--', label='Pickup Escape', color='orange')
plt.loglog(magnetic_m_values_earth_units_F, CF_escape_F, '-', label='Cross-field Escape', color='blue')
plt.loglog(magnetic_m_values_earth_units_F, PC_escape_F, label='Polar Cap Escape', color='purple')
plt.loglog(magnetic_m_values_earth_units_F, CU_escape_F, '--', label='Cusp Escape', color='hotpink')
plt.loglog(magnetic_m_values_earth_units_F, total_escape_F, linewidth=2.5, color='black', label='Total Escape')

plt.xlabel('Magnetic Dipole Moment [Earth Units]', fontsize=14)
plt.ylabel('Escape Rate [Earth Atmospheres per Gyr]', fontsize=14)
plt.title('TRAPPIST-1f Hydrogen Escape vs Magnetic Moment', fontsize=16, weight='bold')

plt.legend(fontsize=11, loc='best', frameon=True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(1e-35, 1e5)
plt.xlim(1e-4, 1e2)

plt.tight_layout()
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import R_earth, M_earth, G, k_B, M_sun


# Same Constants For all Plots 

m_H_G = (3.34 * 10**(-27)) * (u.kg)
magnetic_p_G = 4 * np.pi * 1e-7  * (u.H / u.m) 

T_exo_H_G = 900 * u.K                            
R_G = 1.129 * R_earth.to(u.m)
rg_exo_G = R_G + (396000 * u.m) 
M_G = 1.321 * M_earth.to(u.kg)

ng_sw_G = 1.95e3 * (1/u.cm**3)      
vg_sw_G = 637000 * u.m/u.s                               
prot_mass_G = 1.67e-27 * u.kg

Omega_pc_E = 0.63
r_exo_E = 6871000 * u.m
rg_IMB_G = 7647000 * u.m                           
formfact_o_G = 1.16
Mag_Earth = 8e22 * (u.A * u.m**2)  
mass_G = (1.67 * 10**(-27)) * (u.kg)

magnetic_moments_G = np.logspace(17, 27, 500) * (u.A * u.m**2) 
magnetic_m_values_earth_units_G = magnetic_moments_G / Mag_Earth


# Converting Constant
earth_atmosphere_in_kg_G = (1e-6 * M_earth)  
seconds_per_gyr_G = (1e9 * 365.25 * 24 * 3600)


# Gunell Pickup Escape


Q0_pu_H_G = 5e26 * (1/u.s) 
hg_H_G = (k_B * T_exo_H_G * rg_exo_G**2) / (G * M_G * mass_G)

PU_escape_G = []
pickup_vals_G = []
for M in magnetic_moments_G:
    R_MP_G = ((magnetic_p_G * M**2) / (8 * np.pi**2 * ng_sw_G * vg_sw_G**2 * prot_mass_G))**(1/6)
    if R_MP_G <= rg_exo_G:
        Q_pu_G = Q0_pu_H_G
    else:
        num_G = 2*hg_H_G**3 + 2*hg_H_G**2*R_MP_G + hg_H_G*R_MP_G**2
        den_G = 2*hg_H_G**3 + 2*hg_H_G**2*rg_exo_G + hg_H_G*rg_exo_G**2
        exp_term_G = np.exp((rg_exo_G - R_MP_G) / hg_H_G)
        Q_pu_G = (Q0_pu_H_G * (num_G / den_G) * exp_term_G)
    pickup_vals_G.append(Q_pu_G)
    
for Q_pu_G in pickup_vals_G:
    mass_loss_rate_G = Q_pu_G * mass_G          
    mass_loss_rate_atmospheres_G = (mass_loss_rate_G / earth_atmosphere_in_kg_G) * seconds_per_gyr_G  
    PU_escape_G.append(mass_loss_rate_atmospheres_G.value)  

PU_escape_G = u.Quantity(PU_escape_G)


# Cross-Field Ion Loss


Q0_cf_H_G = 7.7e25 * (1/u.s)     

CF_escape_G = []
cf_vals_G = []
for M in magnetic_moments_G:
    R_MP_G = ((magnetic_p_G * M**2) / (8 * np.pi**2 * ng_sw_G * vg_sw_G**2 * prot_mass_G))**(1/6)
    if R_MP_G >= rg_IMB_G:
        temp_G = 1 - (rg_exo_G / R_MP_G)
        Omega_pc_G = 4 * np.pi * (1 - np.sqrt(temp_G)) if temp_G >= 0 else 0
    else:
        Omega_pc_G = 0
    Q_cf_G = Q0_cf_H_G * ((1 - (Omega_pc_G / (4 * np.pi))) / (1 - (Omega_pc_E / (4 * np.pi))))
    cf_vals_G.append(Q_cf_G)

for Q_cf_G in cf_vals_G:
    mass_loss_rate_G = Q_cf_G * mass_G          
    mass_loss_rate_atmospheres_G = (mass_loss_rate_G / earth_atmosphere_in_kg_G) * seconds_per_gyr_G  
    CF_escape_G.append(mass_loss_rate_atmospheres_G.value) 

CF_escape_G = u.Quantity(CF_escape_G)



# Polar Cap Loss



Q0_pc_H_G = 7.8e25 * (1/u.s)       

PC_escape_G = []
pc_vals_G = []
for M in magnetic_moments_G:
    R_MP_G = ((magnetic_p_G * formfact_o_G**2 * M**2) / (8 * np.pi**2 * ng_sw_G * vg_sw_G**2 * prot_mass_G))**(1/6)
    if R_MP_G > rg_IMB_G:
        temp_G = 1 - (r_exo_E / R_MP_G)
        Omega_pc_G = 4 * np.pi * (1 - np.sqrt(temp_G)) if temp_G >= 0 else 0
    else:
        Omega_pc_G = 0
    Q_pc_G = Q0_pc_H_G * (Omega_pc_G / Omega_pc_E) * ((rg_exo_G / r_exo_E)**2)
    pc_vals_G.append(Q_pc_G)

for Q_pc_G in pc_vals_G:
    mass_loss_rate_G = Q_pc_G * mass_G          
    mass_loss_rate_atmospheres_G = (mass_loss_rate_G / earth_atmosphere_in_kg_G) * seconds_per_gyr_G  
    PC_escape_G.append(mass_loss_rate_atmospheres_G.value)  

PC_escape_G = u.Quantity(PC_escape_G)


# Cusp Escape


Q0_cu_H_G = 5e24 * (1/u.s)           
Qmax_cu_H_G = 5e25 * (1/u.s)         

# Calculate r_c_E for scaling
nsw_G = (1e7) * (1 / u.m**3)  
vsw_G = (604000) * (u.m / u.s)   
r_c_E_G = ((magnetic_p_G * Mag_Earth**2) / (8 * np.pi**2 * nsw_G * vsw_G**2 * prot_mass_G))**(1/6)

CU_escape_G = []
cusp_vals_G = []
for M in magnetic_moments_G:
    R_MP_G = ((magnetic_p_G * M**2) / (8 * np.pi**2 * ng_sw_G * vg_sw_G**2 * prot_mass_G))**(1/6)
    r_c_g = R_MP_G
    if R_MP_G > rg_IMB_G:
        temp_G = 1 - (r_exo_E / R_MP_G)
        Omega_pc_G = 4 * np.pi * (1 - np.sqrt(temp_G)) if temp_G >= 0 else 0
    else:
        Omega_pc_G = 0
    Q_cu_G = min(Q0_cu_H_G * (r_c_g / r_c_E_G)**2, Qmax_cu_H_G) * (Omega_pc_G / Omega_pc_E) * (rg_exo_G / r_exo_E)**2
    cusp_vals_G.append(Q_cu_G)
    
for Q_cu_G in cusp_vals_G:
    mass_loss_rate_G = Q_cu_G * mass_G          
    mass_loss_rate_atmospheres_G = (mass_loss_rate_G / earth_atmosphere_in_kg_G) * seconds_per_gyr_G  
    CU_escape_G.append(mass_loss_rate_atmospheres_G.value)

CU_escape_G = u.Quantity(CU_escape_G)


# Driscoll Ion Pickup Escape


MPU_escape_G = []
magnetic_limited_vals_G = []

time_G = 10 * (u.s)
mass_water_G = 1e21 * (u.kg)
sigma_coll_G = (1 * 10**(-17)) * (u.m**2)
energy_eff_G = (1 / (10.6e44))

for M in magnetic_moments_G:
    r_mp_G = ((magnetic_p_G * M**2) / (8 * np.pi**2 * ng_sw_G * vg_sw_G**2 * prot_mass_G))**(1/6)  
    h_exo_g = (k_B * T_exo_H_G * rg_exo_G**2) / (G * M_G * m_H_G)
    n_exo_g = (1 / (h_exo_g * sigma_coll_G))
    n_L_g = n_exo_g * np.exp((-rg_exo_G / h_exo_g) * (1 - (rg_exo_G / r_mp_G)))
    surface_a_g = (4 * np.pi * r_mp_G**2)
    f_lm_g = (((energy_eff_G * h_exo_g * surface_a_g * (n_L_g - ng_sw_G)) / time_G) * mass_water_G) 
    magnetic_limited_vals_G.append(f_lm_g)

for f_lm_g in magnetic_limited_vals_G:
    mass_loss_rate_G = f_lm_g * mass_G          
    mass_loss_rate_atmospheres_G = (mass_loss_rate_G / earth_atmosphere_in_kg_G) * seconds_per_gyr_G  
    MPU_escape_G.append(mass_loss_rate_atmospheres_G.value)

MPU_escape_G = u.Quantity(MPU_escape_G)


# Energy-Limited Escape


a_G = (84.591 * 84e6) * (u.m)     
M_star_G = 0.09 * M_sun
epsilon_G = 0.1  

# Stellar XUV Luminosity
L_bol_G = 3.828e26 * u.W       
L_xuv_G = 3.4e-4 * L_bol_G       
F_xuv_G = L_xuv_G / (4 * np.pi * a_G**2)                      

rho_G = M_G / ((4/3) * np.pi * R_G**3)                    

r_roche_G = a_G * (M_G / (2 * M_star_G))**(1/3)

K_G = 1 - (3/2) * (R_G / r_roche_G) + (1/2) * (R_G / r_roche_G)**3

mdot_G = epsilon_G * (R_G / R_G)**2 * (3 * F_xuv_G) / (4 * G * rho_G * K_G)
mdot_G = mdot_G.to(u.kg / u.s)

seconds_per_gyr_G = (1e9 * 365.25 * 24 * 3600) * u.s
mdot_earth_per_gyr_G = (mdot_G * seconds_per_gyr_G / M_earth)

mdot_array_G = u.Quantity(np.full(magnetic_moments_G.shape, mdot_earth_per_gyr_G.value), unit=mdot_earth_per_gyr_G.unit)


# Total Escape

total_escape_G = (PU_escape_G.value + CU_escape_G.value + 
                  PC_escape_G.value + CF_escape_G.value)



# Plotting


plt.figure(figsize=(10, 7))

plt.loglog(magnetic_m_values_earth_units_G, mdot_array_G, label='Energy-Limited Escape', color='red', linewidth=3)
plt.loglog(magnetic_m_values_earth_units_G, MPU_escape_G, label='Driscoll Ion Pickup Escape', color='yellow')
plt.loglog(magnetic_m_values_earth_units_G, PU_escape_G, '--', label='Pickup H Escape', color='orange')
plt.loglog(magnetic_m_values_earth_units_G, CF_escape_G, '-', label='Cross-field H Escape', color='blue')
plt.loglog(magnetic_m_values_earth_units_G, PC_escape_G, label='Polar Cap H Escape', color='purple')
plt.loglog(magnetic_m_values_earth_units_G, CU_escape_G, '--', label='Cusp H Escape', color='hotpink')
plt.loglog(magnetic_m_values_earth_units_G, total_escape_G, linewidth=2.5, color='black', label='Total H Escape')

plt.xlabel('Magnetic Dipole Moment [A·m²]')
plt.ylabel('Escape Rate [Earth atmospheres / Gyr]')
plt.title('TRAPPIST-1g Total Hydrogen Escape vs Magnetic Moment')
plt.legend()
plt.ylim(1e-35, 1e5)
plt.xlim(1e-4, 1e2)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import R_earth, M_earth, G, k_B, M_sun



m_H_H = (3.34 * 10**(-27)) * u.kg
magnetic_p_H = 4 * np.pi * 1e-7 * (u.H / u.m) 
prot_mass_H = 1.67e-27 * u.kg
mass_H = (1.67*10**(-27)) * u.kg

T_exo_H = 900 * u.K
R_H_H = 0.755 * R_earth.to(u.m)
rh_exo_H = R_H_H + (396000 * u.m) 
M_H_H = 0.326 * M_earth.to(u.kg)

nh_sw_H = 9.52e2 * (1/u.cm**3)
vh_sw_H = 657000 * u.m/u.s

Omega_pc_E_H = 0.63
r_exo_E_H = 6871000 * u.m
rh_IMB_H = 7647000 * u.m
formfact_o_H = 1.16
Mag_Earth_H = 8e22 * (u.A * u.m**2)  

magnetic_moments_H = np.logspace(17, 27, 500) * (u.A * u.m**2)  
magnetic_m_values_earth_units_H = magnetic_moments_H / Mag_Earth_H

# Conversions
earth_atmosphere_in_kg_H = (1e-6 * M_earth)  
seconds_per_gyr_H = (1e9 * 365.25 * 24 * 3600)


# Pickup Escape (Gunell)


Q0_pu_H_H = 5e26 * (1/u.s) 
hH_H = (k_B * T_exo_H * rh_exo_H**2) / (G * M_H_H * m_H_H)

PU_escape_H = []
for M in magnetic_moments_H:
    R_MP = ((magnetic_p_H * M**2) / (8 * np.pi**2 * nh_sw_H * vh_sw_H**2 * prot_mass_H))**(1/6)
    if R_MP <= rh_exo_H:
        Q_pu = Q0_pu_H_H
    else:
        num = 2*hH_H**3 + 2*hH_H**2*R_MP + hH_H*R_MP**2
        den = 2*hH_H**3 + 2*hH_H**2*rh_exo_H + hH_H*rh_exo_H**2
        exp_term = np.exp((rh_exo_H - R_MP) / hH_H)
        Q_pu = Q0_pu_H_H * (num / den) * exp_term
    mass_loss_rate = Q_pu * mass_H
    PU_escape_H.append((mass_loss_rate / earth_atmosphere_in_kg_H * seconds_per_gyr_H).value)

PU_escape_H = u.Quantity(PU_escape_H)


# Cross-Field Escape


Q0_cf_H_H = 7.7e25 * (1/u.s)

CF_escape_H = []
for M in magnetic_moments_H:
    R_MP = ((magnetic_p_H * M**2) / (8 * np.pi**2 * nh_sw_H * vh_sw_H**2 * prot_mass_H))**(1/6)
    if R_MP >= rh_IMB_H:
        temp = 1 - (rh_exo_H / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cf = Q0_cf_H_H * ((1 - (Omega_pc / (4 * np.pi))) / (1 - (Omega_pc_E_H / (4 * np.pi))))
    mass_loss_rate = Q_cf * mass_H
    CF_escape_H.append((mass_loss_rate / earth_atmosphere_in_kg_H * seconds_per_gyr_H).value)

CF_escape_H = u.Quantity(CF_escape_H)



# Polar Cap Escape


Q0_pc_H_H = 7.8e25 * (1/u.s)

PC_escape_H = []
for M in magnetic_moments_H:
    R_MP = ((magnetic_p_H * formfact_o_H**2 * M**2) / (8 * np.pi**2 * nh_sw_H * vh_sw_H**2 * prot_mass_H))**(1/6)
    if R_MP > rh_IMB_H:
        temp = 1 - (r_exo_E_H / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_pc = Q0_pc_H_H * (Omega_pc / Omega_pc_E_H) * ((rh_exo_H / r_exo_E_H)**2)
    mass_loss_rate = Q_pc * mass_H
    PC_escape_H.append((mass_loss_rate / earth_atmosphere_in_kg_H * seconds_per_gyr_H).value)

PC_escape_H = u.Quantity(PC_escape_H)


# Cusp Escape


Q0_cu_H_H = 5e24 * (1/u.s)  
Qmax_cu_H_H = 5e25 * (1/u.s)

# Earth scaling for cusp
nsw_H = 1e7 / u.m**3
vsw_H = 604000 * u.m/u.s
r_c_E_H = ((magnetic_p_H * Mag_Earth_H**2) / (8 * np.pi**2 * nsw_H * vsw_H**2 * prot_mass_H))**(1/6)

CU_escape_H = []
for M in magnetic_moments_H:
    R_MP = ((magnetic_p_H * M**2) / (8 * np.pi**2 * nh_sw_H * vh_sw_H**2 * prot_mass_H))**(1/6)
    r_c_h = R_MP
    if R_MP > rh_IMB_H:
        temp = 1 - (rh_exo_H / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0

    
    Q_cu = min(Q0_cu_H_H * (r_c_h / r_c_E_H)**2, Qmax_cu_H_H) * (Omega_pc / Omega_pc_E_H) * (rh_exo_H / r_exo_E_H)**2
    mass_loss_rate = Q_cu * mass_H
    CU_escape_H.append((mass_loss_rate / earth_atmosphere_in_kg_H * seconds_per_gyr_H).value)

CU_escape_H = u.Quantity(CU_escape_H)



# Driscoll Ion Pickup Escape


MPU_escape_H = []
time_H = 10 * u.s
mass_water_H = 1e21 * u.kg
sigma_coll_H = 1e-17 * u.m**2
energy_eff_H = 1 / (10.6e44)

for M in magnetic_moments_H:
    r_mp = ((magnetic_p_H * M**2) / (8 * np.pi**2 * nh_sw_H * vh_sw_H**2 * prot_mass_H))**(1/6)  
    h_exo_h = (k_B * T_exo_H * rh_exo_H**2) / (G * M_H_H * m_H_H)
    n_exo_h = 1 / (h_exo_h * sigma_coll_H)
    n_L_h = n_exo_h * np.exp((-rh_exo_H / h_exo_h) * (1 - (rh_exo_H / r_mp)))
    surface_a_h = 4 * np.pi * r_mp**2
    f_lm_h = ((energy_eff_H * h_exo_h * surface_a_h * (n_L_h - nh_sw_H)) / time_H) * mass_water_H
    mass_loss_rate = f_lm_h * mass_H
    MPU_escape_H.append((mass_loss_rate / earth_atmosphere_in_kg_H * seconds_per_gyr_H).value)

MPU_escape_H = u.Quantity(MPU_escape_H)


# Energy-Limited Escape


a_H = (111.817 * 84e6) * u.m   # Semi-major axis
M_star_H = 0.09 * M_sun
epsilon_H = 0.1  

L_bol_H = 3.828e26 * u.W
L_xuv_H = 3.4e-4 * L_bol_H       
F_xuv_H = L_xuv_H / (4 * np.pi * a_H**2)

rho_H = M_H_H / ((4/3) * np.pi * R_H_H**3)                    
r_roche_H = a_H * (M_H_H / (2 * M_star_H))**(1/3)
K_H = 1 - (3/2) * (R_H_H / r_roche_H) + (1/2) * (R_H_H / r_roche_H)**3

mdot_H = epsilon_H * (R_H_H / R_H_H)**2 * (3 * F_xuv_H) / (4 * G * rho_H * K_H)
mdot_H = mdot_H.to(u.kg/u.s)

seconds_per_gyr_H = (1e9 * 365.25 * 24 * 3600) * u.s
mdot_earth_per_gyr_H = (mdot_H * seconds_per_gyr_H / M_earth)
mdot_array_H = u.Quantity(np.full(magnetic_moments_H.shape, mdot_earth_per_gyr_H.value), unit=mdot_earth_per_gyr_H.unit)


# Total Escape

total_escape_H = PU_escape_H.value + CF_escape_H.value + PC_escape_H.value + CU_escape_H.value 


# Plotting

plt.figure(figsize=(10, 7))

ratio = np.divide(total_escape_H, mdot_array_H.value)
    
# Plot ratio
plt.loglog(magnetic_m_values_earth_units, ratio, color='black', linewidth=2.2)
plt.show()
    
plt.figure(figsize=(10, 7))

plt.loglog(magnetic_m_values_earth_units_H, mdot_array_H, label='Energy-Limited Escape', color='red', linewidth=3)
plt.loglog(magnetic_m_values_earth_units_H, MPU_escape_H, label='Driscoll Ion Pickup Escape', color='yellow')
plt.loglog(magnetic_m_values_earth_units_H, PU_escape_H, '--', label='Pickup H Escape', color='orange')
plt.loglog(magnetic_m_values_earth_units_H, CF_escape_H, '-', label='Cross-field H Escape', color='blue')
plt.loglog(magnetic_m_values_earth_units_H, PC_escape_H, label='Polar Cap H Escape', color='purple')
plt.loglog(magnetic_m_values_earth_units_H, CU_escape_H, '--', label='Cusp H Escape', color='hotpink')
plt.loglog(magnetic_m_values_earth_units_H, total_escape_H, linewidth=2.5, color='black', label='Total H Escape')

plt.xlabel('Magnetic Dipole Moment [Earth Units]')
plt.ylabel('Escape Rate [Earth atmospheres per Gyr]')
plt.title('TRAPPIST-1h Total Hydrogen Escape vs Magnetic Moment')
plt.legend()
plt.ylim(1e-35, 1e5)
plt.xlim(1e-4, 1e10)
plt.show()


# In[ ]:





# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def safe_loglog(ax, x, y, **kwargs):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    # Expand scalars or length-1 arrays
    if y.ndim == 0 or y.size == 1:
        y = np.full_like(x, y.item() if y.ndim == 0 else y[0])
    # Replace non-positive values with tiny number
    y = np.where(y <= 0, 1e-30, y)
    # Replace inf and extremely large values
    y = np.where(np.isinf(y) | (y > 1e30), 1e30, y)
    y = np.where(y < 1e-30, 1e-30, y)
    
    mask = np.isfinite(y) & np.isfinite(x) & (y > 0) & (x > 0)
    
    if np.any(mask):
        ax.loglog(x[mask], y[mask], **kwargs)

# Magnetic dipole moments (in Earth units)
magnetic_m_values_earth_units = np.logspace(-2, 4, 500)

# Planet names for TRAPPIST-1 system
planet_names = ["TRAPPIST-1b", "TRAPPIST-1c", "TRAPPIST-1d",
                "TRAPPIST-1e", "TRAPPIST-1f", "TRAPPIST-1g", "TRAPPIST-1h"]

#  escape rate data per planet
escape_sets = [
    (PU_escape_B, CF_escape_B, PC_escape_B, CU_escape_B, MPU_escape_B, Bmdot_earth_per_gyr, total_escape_B),
    (PU_escape_C, CF_escape_C, PC_escape_C, CU_escape_C, MPU_escape_C, mdot_earth_per_gyr_C, total_escape_C),
    (PU_escape_D, CF_escape_D, PC_escape_D, CU_escape_D, MPU_escape_D, mdot_earth_per_gyr_D, total_escape_D),
    (PU_escape_TE, CF_escape_TE, PC_escape_TE, CU_escape_TE, MPU_escape_TE, mdot_earth_per_gyr_TE, total_escape_TE),
    (PU_escape_F, CF_escape_F, PC_escape_F, CU_escape_F, MPU_escape_F, mdot_earth_per_gyr_F, total_escape_F),
    (PU_escape_G, CF_escape_G, PC_escape_G, CU_escape_G, MPU_escape_G, mdot_earth_per_gyr_G, total_escape_G),
    (PU_escape_H, CF_escape_H, PC_escape_H, CU_escape_H, MPU_escape_H, mdot_earth_per_gyr_H, total_escape_H),
]

colors = ['orange', 'blue', 'lawngreen', 'deeppink', 'magenta', 'red', 'black']
linestyles = ['-', '-', '-', '-', '--', '-.', '-']
linewidths = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0]
labels = ['Pickup', 'Cross-field', 'Polar Cap', 'Cusp', 
          'Driscoll Ion Pickup', 'Energy Limited', 'Total']

# Set style for publication quality
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

# Create figure with adjusted spacing
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4, hspace=0.28, wspace=0.25, 
                      left=0.07, right=0.95, top=0.88, bottom=0.10)

axs = []
for i in range(7):
    row = i // 4
    col = i % 4
    ax = fig.add_subplot(gs[row, col])
    axs.append(ax)

# Plot each planet
for i, (ax, name, escapes) in enumerate(zip(axs, planet_names, escape_sets)):
    for y, col, ls, lw, lab in zip(escapes, colors, linestyles, linewidths, labels):
        safe_loglog(ax, magnetic_m_values_earth_units, y, 
                   color=col, linewidth=lw, linestyle=ls, label=lab, alpha=0.9)
    
    # Set explicit axis limits to prevent infinity issues
    ax.set_xlim(5e-2, 1e4)
    ax.set_ylim(1e-5, 1e1)  # Adjust these based on your data range
    
    # Enhanced title with planet letter
    ax.set_title(f'{name}', fontsize=27, pad=10, fontweight='bold')
    
    # Grid styling
    ax.grid(True, which='major', alpha=0.4, linewidth=0.6)
    ax.grid(True, which='minor', alpha=0.2, linewidth=0.5)


    
    # Tick parameters
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=17)
    ax.tick_params(which='major', length=11, width=1.5)
    ax.tick_params(which='minor', length=6, width=1)

# Create a single legend for all subplots at the empty position
legend_ax = fig.add_subplot(gs[1, 3])
legend_ax.axis('off')

# Create dummy lines for legend
handles = []
for col, ls, lw, lab in zip(colors, linestyles, linewidths, labels):
    line = plt.Line2D([0], [0], color=col, linewidth=lw, linestyle=ls, label=lab)
    handles.append(line)

legend_ax.legend(handles=handles, loc='center', fontsize=18, 
                frameon=True, fancybox=True, shadow=True,
                title='Escape Mechanisms', title_fontsize=21)

# Add overall x and y labels
fig.text(0.53, 0.02, "Magnetic Dipole Moment [M$_E$]", 
         ha='center', fontsize=29)
fig.text(0.02, 0.5, "Escape Rate [Earth atm / Gyr]", 
         va='center', rotation='vertical', fontsize=29)

# Add a main title
#fig.suptitle('Atmospheric Escape Rates vs Magnetic Dipole Moment\nTRAPPIST-1 System', 
#             fontsize=32, y=0.97)

plt.savefig('trappist1_escape_rates.pdf', bbox_inches='tight')
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def safe_semilogy(ax, x, y, **kwargs):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    y = np.where(y <= 0, 1e-30, y)
    y = np.where(np.isinf(y) | (y > 1e30), 1e30, y)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if np.any(mask):
        ax.semilogy(x[mask], y[mask], **kwargs)

magnetic_m_values_earth_units = np.logspace(-2, 4, 500)

planet_names = ["TRAPPIST-1b", "TRAPPIST-1c", "TRAPPIST-1d",
                "TRAPPIST-1e", "TRAPPIST-1f", "TRAPPIST-1g", "TRAPPIST-1h"]

escape_sets = [
    (PU_escape_B, CF_escape_B, PC_escape_B, CU_escape_B, MPU_escape_B, Bmdot_earth_per_gyr, total_escape_B),
    (PU_escape_C, CF_escape_C, PC_escape_C, CU_escape_C, MPU_escape_C, mdot_earth_per_gyr_C, total_escape_C),
    (PU_escape_D, CF_escape_D, PC_escape_D, CU_escape_D, MPU_escape_D, mdot_earth_per_gyr_D, total_escape_D),
    (PU_escape_TE, CF_escape_TE, PC_escape_TE, CU_escape_TE, MPU_escape_TE, mdot_earth_per_gyr_TE, total_escape_TE),
    (PU_escape_F, CF_escape_F, PC_escape_F, CU_escape_F, MPU_escape_F, mdot_earth_per_gyr_F, total_escape_F),
    (PU_escape_G, CF_escape_G, PC_escape_G, CU_escape_G, MPU_escape_G, mdot_earth_per_gyr_G, total_escape_G),
    (PU_escape_H, CF_escape_H, PC_escape_H, CU_escape_H, MPU_escape_H, mdot_earth_per_gyr_H, total_escape_H),
]



plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5


fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4, hspace=0.28, wspace=0.25,
                      left=0.07, right=0.95, top=0.88, bottom=0.10)

axs = []
for i in range(7):
    row = i // 4
    col = i % 4
    ax = fig.add_subplot(gs[row, col])
    axs.append(ax)


for i, (ax, name, escapes) in enumerate(zip(axs, planet_names, escape_sets)):
    PU, CF, PC, CU, MPU, energy_lim, total = escapes
    
    # Compute ratio 
    ratio = np.divide(total, energy_lim.value)
    
    ax.loglog(magnetic_m_values_earth_units, ratio, color='black', linewidth=2.2)
    
    ax.axhline(1, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
    
    ax.set_xlim(1e0, 1e4)
    ax.set_ylim(1e-4, 1e0)
    ax.set_title(f'{name}', fontsize=27, pad=10, fontweight='bold')
    ax.grid(True, which='major', alpha=0.4, linewidth=0.6)
    ax.grid(True, which='minor', alpha=0.2, linewidth=0.5)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=17)
    ax.tick_params(which='major', length=11, width=1.5)
    ax.tick_params(which='minor', length=6, width=1)



fig.text(0.53, 0.02, "Magnetic Dipole Moment [M$_E$]", ha='center', fontsize=29)
fig.text(0.02, 0.5, r"Total Magnetic-Limited / Energy-Limited Escape Rate", 
         va='center', rotation='vertical', fontsize=29)

plt.savefig('trappist1_ratio_grid_black.pdf', bbox_inches='tight')
plt.show()


# In[ ]:


# check

