#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import R_earth, M_earth
from scipy.interpolate import interp1d


# Same Constants For all Plots 

m_H = 1.6735575e-27
magnetic_p = 4 * np.pi * 1e-7  # H/m
k = 1.3e-23
G = 6.7e-11

T_exo_H = 900
R_F = 1.045 * R_earth.to(u.m).value
rf_exo = R_F + 396000
M_F = 1.039 * M_earth.to(u.kg).value

nf_sw = 2.99 * 1e6  # convert cm^-3 to m^-3
vf_sw = 624000  # m/s
prot_mass = 1.67e-27

Omega_pc_E = 0.63
r_exo_E = 6871000
rf_IMB = 7647000
formfact_o = 1.16
Mag_Earth = 7.77e22

magnetic_moments = np.logspace(17, 27, 500)


# Pickup Escape

Q0_pu_H = 5e26
hf_H = (k * T_exo_H * rf_exo**2) / (G * M_F * m_H)

pickup_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6)
    if R_MP <= rf_exo:
        Q_pu = Q0_pu_H
    else:
        num = 2*hf_H**3 + 2*hf_H**2*R_MP + hf_H*R_MP**2
        den = 2*hf_H**3 + 2*hf_H**2*rf_exo + hf_H*rf_exo**2
        exp_term = np.exp((rf_exo - R_MP) / hf_H)
        Q_pu = Q0_pu_H * (num / den) * exp_term
    pickup_vals.append(Q_pu)
pickup_vals = np.array(pickup_vals)


# Cross-Field Ion Loss

Q0_cf_H = 7.7e25

cf_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6)
    if R_MP >= rf_IMB:
        temp = 1 - (rf_exo / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cf = Q0_cf_H * ((1 - (Omega_pc / (4 * np.pi))) / (1 - (Omega_pc_E / (4 * np.pi))))
    cf_vals.append(Q_cf)
cf_vals = np.array(cf_vals)


# Polar Cap Loss

Q0_pc_H = 7.8e25

pc_vals = []
for M in magnetic_moments:
    R_MP = (((magnetic_p * formfact_o**2 * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6))
    if R_MP > rf_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_pc = Q0_pc_H * (Omega_pc / Omega_pc_E) * ((rf_exo / r_exo_E)**2)
    pc_vals.append(Q_pc)
pc_vals = np.array(pc_vals)


# Cusp Escape

Q0_cu_H = 5e24
Qmax_cu_H = 5e25

# Calculate r_c_E for scaling
nsw = 6e6
vsw = 4e5
r_c_E = ((magnetic_p * Mag_Earth**2) / (8 * np.pi**2 * nsw * vsw**2 * prot_mass))**(1/6)

cusp_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6)
    r_c_f = R_MP
    if R_MP > rf_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cu = min(Q0_cu_H * (r_c_f / r_c_E)**2, Qmax_cu_H) * (Omega_pc / Omega_pc_E) * (rf_exo / r_exo_E)**2
    cusp_vals.append(Q_cu)
cusp_vals = np.array(cusp_vals)


# Calculate energy-limited escape rate

# Energy limited escape
I_XUV_values = 0.85 #*(u.W/u.m**2)  

#2935.128486  #* erg cm**(−2)* s**(−1)
#0.85 *(u.W/u.m**2)  

K = 0.55  

Gamma = (R_F * (rf_exo**2) * I_XUV_values) / (M_F * G * K)  
#Gamma_atmospheres = ((Gamma / earth_atmosphere_in_kg) * seconds_per_gyr).decompose() 

# Create an array of constant escape rates
Gamma_array = np.full_like(magnetic_moments, Gamma)

# Total Escape  
total_escape = pickup_vals + cf_vals + pc_vals + cusp_vals + Gamma_array


plt.figure(figsize=(10, 7))

plt.loglog(magnetic_moments, Gamma_array, label='Energy Limited Escape', color='red', linewidth=3)
plt.loglog(magnetic_moments, pickup_vals, '--', label='Pickup H Escape', color='orange')
plt.loglog(magnetic_moments, cf_vals, '-', label='Cross-field H Escape', color='blue')
plt.loglog(magnetic_moments, pc_vals, label='Polar Cap H Escape', color='purple')
plt.loglog(magnetic_moments, cusp_vals, '--', label='Cusp H Escape', color='hotpink')
plt.loglog(magnetic_moments, total_escape, linewidth=2.5, color='black', label='Total H Escape')


plt.xlabel('Magnetic Dipole Moment [A·m²]')
plt.ylabel('Escape Rate [s⁻¹]')
plt.title('TRAPPIST-1f Total Hydrogen Escape vs Magnetic Moment')
plt.legend()
plt.ylim(1e3, 1e28)
plt.xlim(1e18, 1e27)
plt.show()


# In[21]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import R_earth, M_earth, G, k_B, M_sun
from scipy.interpolate import interp1d


# Same Constants For all Plots 

m_H = (3.34 * 10**(-27)) * (u.kg)
magnetic_p = 4 * np.pi * 1e-7  * (u.H / u.m) 


T_exo_H = 900 * u.K                            # DIFFERENT (USED 1107 BEFORE)
R_F = 1.045 * R_earth.to(u.m)
rf_exo = R_F + (396000 *u.m) 
M_F = 1.039 * M_earth.to(u.kg)

nf_sw = 2.99e3 *(1/u.cm**3)      # * 1e6 convert cm^-3 to m^-3   # DIFFERENT (USED EARTH PARAMS)
vf_sw = 624000  * u.m/u.s                               # DIFFERENT (USED EARTH PARAMS)
prot_mass = 1.67e-27 *u.kg

Omega_pc_E = 0.63
r_exo_E = 6871000 *u.m
rf_IMB = 7647000  *u.m                           # DIFFERENT (USED 6666800 m)
formfact_o = 1.16
Mag_Earth = 8e22    * (u.A * u.m**2)  
mass_h = (3.34 * 10**(-27)) * u.kg

magnetic_moments = np.logspace(17, 27, 500)   * (u.A * u.m**2)  


# Pickup Escape

Q0_pu_H = 5e26 *( 1/u.s) 
hf_H = (k_B * T_exo_H * rf_exo**2) / (G * M_F * m_H)

pickup_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6)
    if R_MP <= rf_exo:
        Q_pu = Q0_pu_H
    else:
        num = 2*hf_H**3 + 2*hf_H**2*R_MP + hf_H*R_MP**2
        den = 2*hf_H**3 + 2*hf_H**2*rf_exo + hf_H*rf_exo**2
        exp_term = np.exp((rf_exo - R_MP) / hf_H)
        Q_pu = (Q0_pu_H * (num / den) * exp_term)
    pickup_vals.append(Q_pu)
pickup_vals = u.Quantity(pickup_vals)


# Cross-Field Ion Loss

Q0_cf_H = 7.7e25   *( 1/u.s)     # Earth Params

cf_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6)
    if R_MP >= rf_IMB:
        temp = 1 - (rf_exo / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cf = Q0_cf_H * ((1 - (Omega_pc / (4 * np.pi))) / (1 - (Omega_pc_E / (4 * np.pi))))
    cf_vals.append(Q_cf)
cf_vals = u.Quantity(cf_vals)


# Polar Cap Loss

Q0_pc_H = 7.8e25  *( 1/u.s)       # Earth Params

pc_vals = []
for M in magnetic_moments:
    R_MP = (((magnetic_p * formfact_o**2 * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6))
    if R_MP > rf_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_pc = Q0_pc_H * (Omega_pc / Omega_pc_E) * ((rf_exo / r_exo_E)**2)
    pc_vals.append(Q_pc)
pc_vals = u.Quantity(pc_vals)


# Cusp Escape

Q0_cu_H = 5e24     *( 1/u.s)           # Earth Params
Qmax_cu_H = 5e25   *( 1/u.s)         # Earth Params

# Calculate r_c_E for scaling
nsw = (1e7) * (1 / u.m**3)  
vsw = (604000) * (u.m / u.s)   
r_c_E = ((magnetic_p * Mag_Earth**2) / (8 * np.pi**2 * nsw * vsw**2 * prot_mass))**(1/6)

cusp_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6)
    r_c_f = R_MP
    if R_MP > rf_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cu = min(Q0_cu_H * (r_c_f / r_c_E)**2, Qmax_cu_H) * (Omega_pc / Omega_pc_E) * (rf_exo / r_exo_E)**2
    cusp_vals.append(Q_cu)
cusp_vals = u.Quantity(cusp_vals)


# Calculate Peter's (Ion Pickup Escape)

magnetic_limited_vals = []

time = 10  * (u.s)
mass_water = 1e21 * (u.kg)
sigma_coll = (1 * 10**(-17))  * (u.m**2)
energy_eff = (1 / (10.6e44))

for M in magnetic_moments:
    r_mp = (((magnetic_p * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6))  
    h_exo_f = (k_B * T_exo_H * rf_exo**2) / (G * M_F * mass_h)
    n_exo_f = (1 / (h_exo_f * sigma_coll))  #.to(1 / u.m**3)
    n_L_f = n_exo_f * np.exp((-rf_exo / h_exo_f) * (1 - (rf_exo / r_mp)))
    surface_a_f = (4 * np.pi * r_mp**2)
    f_lm_f = (((energy_eff * h_exo_f * surface_a_f * (n_L_f - nf_sw)) / time) * mass_water) #.decompose()
    magnetic_limited_vals.append(f_lm_f)

magnetic_limited_vals =u.Quantity(magnetic_limited_vals)



# Calculate energy-limited escape rate

#I_XUV_values = 0.85 #*(u.W/u.m**2)    # DIFFERENT (USED 0.48)

a = (69.543 * 84e6) * (u.m)     # Used ratio from Eric Agol paper ( a/R*)
print(a)

L_xuv = ((1.77 * 10**(-7)) *  3.83e26) * (u.W) 

print(L_xuv)

M_star = 0.09*M_sun

r_RL = (((M_F/(3*M_star))**(1/3)) * a)   #Roche Lobe

eta = r_RL/R_F

K = ((((eta.value-1)**2) * ((2*eta.value)+1))/(2*eta.value**3))    # Roche lobe correction factor K

I_XUV_values = (L_xuv / (4*np.pi * (a**2)))        # XUV flux

Gamma = (R_F * (rf_exo**2) * I_XUV_values) / (M_F * G * K * mass_h)  

#Gamma_atmospheres = ((Gamma / earth_atmosphere_in_kg) * seconds_per_gyr).decompose() 

Gamma_array = u.Quantity(np.full(magnetic_moments.shape, Gamma.value), unit=Gamma.unit)


# Total Escape  
total_escape = pickup_vals.value + cf_vals.value + pc_vals.value + cusp_vals.value  

plt.figure(figsize=(10, 7))

plt.loglog(magnetic_moments,  magnetic_limited_vals, label = 'Driscoll Ion Pickup Escape', color = 'yellow')
plt.loglog(magnetic_moments, Gamma_array, label='Energy Limited Escape', color='red', linewidth=3)
plt.loglog(magnetic_moments, pickup_vals, '--', label='Pickup H Escape', color='orange')
plt.loglog(magnetic_moments, cf_vals, '-', label='Cross-field H Escape', color='blue')
plt.loglog(magnetic_moments, pc_vals, label='Polar Cap H Escape', color='purple')
plt.loglog(magnetic_moments, cusp_vals, '--', label='Cusp H Escape', color='hotpink')
plt.loglog(magnetic_moments, total_escape, linewidth=2.5, color='black', label='Total H Escape')


plt.xlabel('Magnetic Dipole Moment [A·m²]')
plt.ylabel('Escape Rate [s⁻¹]')
plt.title('TRAPPIST-1f Total Hydrogen Escape vs Magnetic Moment')
plt.legend()
plt.ylim(1e1, 1e35)
plt.xlim(2e19, 6e24)
plt.show()


# In[25]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import R_earth, M_earth, G, k_B, M_sun
from scipy.interpolate import interp1d


# Same Constants For all Plots 

m_H = (3.34 * 10**(-27)) * (u.kg)
magnetic_p = 4 * np.pi * 1e-7  * (u.H / u.m) 


T_exo_H = 900 * u.K                            # DIFFERENT (USED 1107 BEFORE)
R_F = 1.045 * R_earth.to(u.m)
M_F = 1.039 * M_earth.to(u.kg)

nf_sw = 2.99e3 *(1/u.cm**3)      # * 1e6 convert cm^-3 to m^-3   # DIFFERENT (USED EARTH PARAMS)
vf_sw = 624000  * u.m/u.s                               # DIFFERENT (USED EARTH PARAMS)
prot_mass = 1.67e-27 *u.kg

Omega_pc_E = 0.63
r_exo_E = 6871000 *u.m
rf_IMB = 7647000  *u.m                           # DIFFERENT (USED 6666800 m)
formfact_o = 1.16
Mag_Earth = 8e22    * (u.A * u.m**2)  
mass_h = (3.34 * 10**(-27)) * u.kg
mass = (1.67*10**(-27)) * (u.kg)
z_exo = 396 * (u.km)
rf_exo = R_F + z_exo 

magnetic_moments = np.logspace(17, 27, 500)   * (u.A * u.m**2)  
magnetic_m_values_earth_units = magnetic_moments / Mag_Earth



# Converting Constant
earth_atmosphere_in_kg = (1e-6 * M_earth)  
seconds_per_gyr = (1e9 * 365.25 * 24 * 3600)

# Calculate Peter's (Ion Pickup Escape)
MPU_escape = []
magnetic_limited_vals = []

time = 10  * (u.s)
mass_water = 1e21 * (u.kg)
sigma_coll = (1 * 10**(-17))  * (u.m**2)
energy_eff = (1 / (10.6e44))



for M in magnetic_moments:
    r_mp = (((magnetic_p * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6))  
    h_exo_f = (k_B * T_exo_H * rf_exo**2) / (G * M_F * mass_h)
    n_exo_f = (1 / (h_exo_f * sigma_coll))  #.to(1 / u.m**3)
    n_L_f = n_exo_f * np.exp((-rf_exo / h_exo_f) * (1 - (rf_exo / r_mp)))
    surface_a_f = (4 * np.pi * r_mp**2)
    f_lm_f = (((energy_eff * h_exo_f * surface_a_f * (n_L_f - nf_sw)) / time) * mass_water) #.decompose()
    magnetic_limited_vals.append(f_lm_f)

for f_lm_f in magnetic_limited_vals:
    mass_loss_rate =  f_lm_f * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    MPU_escape.append(mass_loss_rate_atmospheres.value)


MPU_escape =u.Quantity(MPU_escape)


# Gunell Pickup Escape

Q0_pu_H = 5e26 *( 1/u.s) 
hf_H = (k_B * T_exo_H * rf_exo**2) / (G * M_F * m_H)

PU_escape = []
pickup_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6)
    if R_MP <= rf_exo:
        Q_pu = Q0_pu_H
    else:
        num = 2*hf_H**3 + 2*hf_H**2*R_MP + hf_H*R_MP**2
        den = 2*hf_H**3 + 2*hf_H**2*rf_exo + hf_H*rf_exo**2
        exp_term = np.exp((rf_exo - R_MP) / hf_H)
        Q_pu = (Q0_pu_H * (num / den) * exp_term)
    pickup_vals.append(Q_pu)

for Q_pu in pickup_vals:
    mass_loss_rate =  Q_pu * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    PU_escape.append(mass_loss_rate_atmospheres.value)  


PU_escape = u.Quantity(PU_escape)


# Cross-Field Ion Loss

Q0_cf_H = 7.7e25   *( 1/u.s)     # Earth Params

CF_escape = []
cf_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6)
    if R_MP >= rf_IMB:
        temp = 1 - (rf_exo / R_MP)
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

Q0_pc_H = 7.8e25  *( 1/u.s)       # Earth Params

PC_escape = []
pc_vals = []
for M in magnetic_moments:
    R_MP = (((magnetic_p * formfact_o**2 * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6))
    if R_MP > rf_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_pc = Q0_pc_H * (Omega_pc / Omega_pc_E) * ((rf_exo / r_exo_E)**2)
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
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nf_sw * vf_sw**2 * prot_mass))**(1/6)
    r_c_f = R_MP
    if R_MP > rf_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cu = min(Q0_cu_H * (r_c_f / r_c_E)**2, Qmax_cu_H) * (Omega_pc / Omega_pc_E) * (rf_exo / r_exo_E)**2
    cusp_vals.append(Q_cu)


for Q_cu in cusp_vals:
    mass_loss_rate =  Q_cu * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    CU_escape.append(mass_loss_rate_atmospheres.value)

CU_escape = u.Quantity(CU_escape)


# Calculate energy-limited escape rate


a = (69.543 * 84e6) * (u.m)     # Used ratio from Eric Agol paper ( a/R*)
M_star = 0.09*M_sun
epsilon = 0.1  # heating efficiency


# Stellar XUV Luminosity
L_bol = 3.828e26 * u.W       # solar luminosity
L_xuv = 3.4e-4 * L_bol       
F_xuv = L_xuv / (4 * np.pi * a**2)                      # flux at planet

# Its labeled as the mass density of the plnet. I used the formula to calculate density: mass/V  
rho = M_F / ((4/3) * np.pi * R_F**3)                    

# Roche lobe radius (Becker+2020 uses 2 in denominator!)
r_roche = a * (M_F / (2 * M_star))**(1/3)

# Tidal enhancement factor K
K = 1 - (3/2) * (R_F / r_roche) + (1/2) * (R_F / r_roche)**3

# Energy-limited escape rate (mass loss) in kg/s
mdot = epsilon * (R_F / R_F)**2 * (3 * F_xuv) / (4 * G * rho * K)
mdot = mdot.to(u.kg / u.s)

seconds_per_gyr = (1e9 * 365.25 * 24 * 3600) * u.s
mdot_earth_per_gyr = (mdot * seconds_per_gyr / M_earth)


mdot_array = u.Quantity(np.full(magnetic_moments.shape, mdot_earth_per_gyr.value), unit=mdot_earth_per_gyr.unit)




# Total Escape  
total_escape = PU_escape.value + CU_escape.value + PC_escape.value + CF_escape.value

plt.figure(figsize=(10, 7))

plt.loglog(magnetic_m_values_earth_units, mdot_array, color='red', linewidth=3, label='Energy-Limited Mass Loss')
plt.loglog(magnetic_m_values_earth_units,  MPU_escape, label = 'Driscoll Ion Pickup Escape', color = 'yellow')
plt.loglog(magnetic_m_values_earth_units, PU_escape, '--', label='Pickup H Escape', color='orange')
plt.loglog(magnetic_m_values_earth_units, CF_escape, '-', label='Cross-field H Escape', color='blue')
plt.loglog(magnetic_m_values_earth_units, PC_escape, label='Polar Cap H Escape', color='purple')
plt.loglog(magnetic_m_values_earth_units, CU_escape, '--', label='Cusp H Escape', color='hotpink')
plt.loglog(magnetic_m_values_earth_units, total_escape, linewidth=2.5, color='black', label='Total H Escape')


plt.xlabel('Magnetic Dipole Moment [A·m²]')
plt.ylabel('Escape Rate [s⁻¹]')
plt.title('TRAPPIST-1f Total Hydrogen Escape vs Magnetic Moment')
plt.legend()
plt.ylim(1e-35, 1e5)
plt.xlim(1e-4, 1e2)

#plt.ylim(1e-5, 1e0)
#plt.xlim(1e2, 1e4)
plt.show()


# In[ ]:




