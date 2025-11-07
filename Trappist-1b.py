#!/usr/bin/env python
# coding: utf-8

# In[55]:


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
R_B = 1.116 * R_earth.to(u.m).value
rb_exo = R_B + 396000
M_B = 1.374 * M_earth.to(u.kg).value

nb_sw = 6.59e4 * 1e6  # convert cm^-3 to m^-3
vb_sw = 470000  # m/s
prot_mass = 1.67e-27

Omega_pc_E = 0.63
r_exo_E = 6871000
rb_IMB = 7647000
formfact_o = 1.16
Mag_Earth = 7.77e22

magnetic_moments = np.logspace(17, 27, 500)


# Pickup Escape

Q0_pu_H = 5e26
hb_H = (k * T_exo_H * rb_exo**2) / (G * M_B * m_H)

pickup_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    if R_MP <= rb_exo:
        Q_pu = Q0_pu_H
    else:
        num = 2*hb_H**3 + 2*hb_H**2*R_MP + hb_H*R_MP**2
        den = 2*hb_H**3 + 2*hb_H**2*rb_exo + hb_H*rb_exo**2
        exp_term = np.exp((rb_exo - R_MP) / hb_H)
        Q_pu = Q0_pu_H * (num / den) * exp_term
    pickup_vals.append(Q_pu)
pickup_vals = np.array(pickup_vals)


# Cross-Field Ion Loss

Q0_cf_H = 7.7e25

cf_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    if R_MP > rb_IMB:
        temp = 1 - (rb_exo / R_MP)
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
    R_MP = (((magnetic_p * formfact_o**2 * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6))
    if R_MP > rb_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_pc = Q0_pc_H * (Omega_pc / Omega_pc_E) * ((rb_exo / r_exo_E)**2)
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
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    r_c_b = R_MP
    if R_MP > rb_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cu = min(Q0_cu_H * (r_c_b / r_c_E)**2, Qmax_cu_H) * (Omega_pc / Omega_pc_E) * (rb_exo / r_exo_E)**2
    cusp_vals.append(Q_cu)
cusp_vals = np.array(cusp_vals)


# Total Escape  
total_escape = pickup_vals + cf_vals + pc_vals + cusp_vals


plt.figure(figsize=(10, 7))

plt.loglog(magnetic_moments, pickup_vals, '--', label='Pickup H Escape', color='orange')
plt.loglog(magnetic_moments, cf_vals, '-', label='Cross-field H Escape', color='blue')
plt.loglog(magnetic_moments, pc_vals, label='Polar Cap H Escape', color='purple')
plt.loglog(magnetic_moments, cusp_vals, '--', label='Cusp H Escape', color='hotpink')
plt.loglog(magnetic_moments, total_escape, linewidth=2.5, color='black', label='Total H Escape')


plt.xlabel('Magnetic Dipole Moment [A·m²]')
plt.ylabel('Escape Rate [s⁻¹]')
plt.title('TRAPPIST-1b Total Hydrogen Escape vs Magnetic Moment')
plt.legend()
plt.ylim(1e23, 1e28)
plt.xlim(1e20, 1e27)
plt.show()


# In[7]:


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

# Converting Constant
earth_atmosphere_in_kg = (1e-6 * M_earth)  
seconds_per_gyr = 1e9 * 365.25 * 24 * 3600




# Gunnel Pickup Escape
 
Q0_pu_H = 5e26   * (1/u.s)
hb_H = (k_B * T_exo_H * rb_exo**2) / (G * M_B * mass)

pickup_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    if R_MP <= rb_exo:
        Q_pu = Q0_pu_H
    else:
        num = 2*hb_H**3 + 2*hb_H**2*R_MP + hb_H*R_MP**2
        den = 2*hb_H**3 + 2*hb_H**2*rb_exo + hb_H*rb_exo**2
        exp_term = np.exp((rb_exo - R_MP) / hb_H)
        Q_pu = Q0_pu_H * (num / den) * exp_term
    pickup_vals.append(Q_pu)
pickup_vals = u.Quantity(pickup_vals)


# Cross-Field Ion Loss

Q0_cf_H = 7.7e25   * (1/u.s)

cf_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    if R_MP >= rb_IMB:
        temp = 1 - (rb_exo / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cf = Q0_cf_H * ((1 - (Omega_pc / (4 * np.pi))) / (1 - (Omega_pc_E / (4 * np.pi))))
    cf_vals.append(Q_cf)
cf_vals = u.Quantity(cf_vals)


# Polar Cap Loss

Q0_pc_H = 7.8e25     * (1/u.s)

pc_vals = []
for M in magnetic_moments:
    R_MP = (((magnetic_p * formfact_o**2 * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6))
    if R_MP > rb_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_pc = Q0_pc_H * (Omega_pc / Omega_pc_E) * ((rb_exo / r_exo_E)**2)
    pc_vals.append(Q_pc)
pc_vals = u.Quantity(pc_vals)

  


# Cusp Escape

Q0_cu_H = 5e24     *( 1/u.s)           
Qmax_cu_H = 5e25   *( 1/u.s) 

# Calculate r_c_E for scaling
nsw = (1e7) * (1 / u.m**3)  
vsw = (604000) * (u.m / u.s)   
r_c_E = ((magnetic_p * Mag_Earth**2) / (8 * np.pi**2 * nsw * vsw**2 * prot_mass))**(1/6)

cusp_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    r_c_b = R_MP
    if R_MP > rb_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cu = min(Q0_cu_H * (r_c_b / r_c_E)**2, Qmax_cu_H) * (Omega_pc / Omega_pc_E) * (rb_exo / r_exo_E)**2
    cusp_vals.append(Q_cu)
cusp_vals = u.Quantity(cusp_vals)


# Peter Driscoll's Magnetic-Limited Escape 

magnetic_limited_vals = []

time = 10  * (u.s)
mass_water = 1e21 * (u.kg)
sigma_coll = (1 * 10**(-17))  * (u.m**2)
energy_eff = (1 / (10.6e44))

for M in magnetic_moments:
    r_mp = (((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6))  
    h_exo_b = (k_B * T_exo_H * rb_exo**2) / (G * M_B * mass)
    n_exo_b = (1 / (h_exo_b * sigma_coll))  #.to(1 / u.m**3)
    n_L_b = n_exo_b * np.exp((-rb_exo / h_exo_b) * (1 - (rb_exo / r_mp)))
    surface_a_b = (4 * np.pi * r_mp**2)
    f_lm_b = (((energy_eff * h_exo_b * surface_a_b * (n_L_b - nb_sw)) / time) * mass_water) #.decompose()
    magnetic_limited_vals.append(f_lm_b)

magnetic_limited_vals =u.Quantity(magnetic_limited_vals)



# Calculate energy-limited escape rate

#I_XUV_values = 0.85 #*(u.W/u.m**2)    # DIFFERENT (USED 0.48)

a = (20.843 * 84e6) * (u.m)     # Used ratio from Eric Agol paper ( a/R*)
L_xuv = ((1.77 * 10**(-7)) *  3.83e26) * (u.W) 

M_star = 0.09*M_sun

r_RL = (((M_B/(3*M_star))**(1/3)) * a)   #Roche Lobe

eta = r_RL/R_B

K = ((((eta.value-1)**2) * ((2*eta.value)+1))/(2*eta.value**3))    # Roche lobe correction factor K

I_XUV_values = (L_xuv / (4*np.pi * (a**2)))        # XUV flux

Gamma = (R_B * (rb_exo**2) * I_XUV_values) / (M_B * G * K * m_H)  

#Gamma_atmospheres = ((Gamma / earth_atmosphere_in_kg) * seconds_per_gyr).decompose() 

Gamma_array = u.Quantity(np.full(magnetic_moments.shape, Gamma.value), unit=Gamma.unit)


# Total Escape  
total_escape = pickup_vals.value + cf_vals.value + pc_vals.value + cusp_vals.value  




# Convert magnetic moments to Earth units for plotting

magnetic_m_values_earth_units = magnetic_moments / Mag_Earth





plt.figure(figsize=(10, 7))

#plt.loglog(magnetic_moments, Gamma_array, label='Energy Limited Escape', color='red', linewidth=3)
plt.loglog(magnetic_moments,  magnetic_limited_vals, label = 'Driscoll Ion Pickup Escape', color = 'yellow')
#plt.loglog(magnetic_moments, pickup_vals, '--', label='Pickup H Escape', color='orange')
#plt.loglog(magnetic_moments, cf_vals, '-', label='Cross-field H Escape', color='blue')
#plt.loglog(magnetic_moments, PC_escape, label='Polar Cap H Escape', color='purple')
#plt.loglog(magnetic_moments, cusp_vals, '--', label='Cusp H Escape', color='hotpink')
#plt.loglog(magnetic_moments, total_escape, linewidth=2.5, color='black', label='Total H Escape')


plt.xlabel('Magnetic Dipole Moment [A·m²]')
plt.ylabel('Escape Rate [s⁻¹]')
plt.title('TRAPPIST-1b Total Hydrogen Escape vs Magnetic Moment')
plt.legend()
plt.ylim(1e2, 1e35)
plt.xlim(1e18, 1e27)
plt.show()


# In[44]:


r_mp_vals = []
n_L_vals = []

for M in magnetic_moments:
    r_mp = (((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6))  
    h_exo_b = (k_B * T_exo_H * rb_exo**2) / (G * M_B * mass)
    n_exo_b = (1 / (h_exo_b * sigma_coll))  
    exponent = (-rb_exo / h_exo_b) * (1 - (rb_exo / r_mp))
    n_L = n_exo_b * np.exp(exponent)

    r_mp_vals.append(r_mp.to(u.km).value)
    n_L_vals.append(n_L.to(1/u.m**3).value)

# Plotting
fig, ax1 = plt.subplots(figsize=(10,6))

color = 'tab:blue'
ax1.set_xlabel('Magnetic Dipole Moment [A·m²]')
ax1.set_ylabel('Magnetopause Radius [km]', color=color)
ax1.loglog(magnetic_moments, r_mp_vals, color=color, label='Magnetopause Radius')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('n_L at r_mp [1/m³]', color=color)
ax2.loglog(magnetic_moments, n_L_vals, color=color, linestyle='--', label='Exospheric H Density at r_mp')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Magnetopause Radius and Exospheric H Density at r_mp vs Magnetic Moment')
fig.tight_layout()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.show()


# In[10]:


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
hb_H = (k_B * T_exo_H * rb_exo**2) / (G * M_B * m_H)

PU_escape = []
pickup_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    if R_MP <= rb_exo:
        Q_pu = Q0_pu_H
    else:
        num = 2*hb_H**3 + 2*hb_H**2*R_MP + hb_H*R_MP**2
        den = 2*hb_H**3 + 2*hb_H**2*rb_exo + hb_H*rb_exo**2
        exp_term = np.exp((rb_exo - R_MP) / hb_H)
        Q_pu = Q0_pu_H * (num / den) * exp_term
    pickup_vals.append(Q_pu)

for Q_pu in pickup_vals:
    mass_loss_rate =  Q_pu * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    PU_escape.append(mass_loss_rate_atmospheres.value)  
PU_escape = u.Quantity(PU_escape)



# Cross-Field Ion Loss

Q0_cf_H = 7.7e25   * (1/u.s)

CF_escape = []
cf_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    if R_MP >= rb_IMB:
        temp = 1 - (rb_exo / R_MP)
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

Q0_pc_H = 7.8e25     * (1/u.s)

PC_escape = []
pc_vals = []
for M in magnetic_moments:
    R_MP = (((magnetic_p * formfact_o**2 * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6))
    if R_MP > rb_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_pc = Q0_pc_H * (Omega_pc / Omega_pc_E) * ((rb_exo / r_exo_E)**2)
    pc_vals.append(Q_pc)


for Q_pc in pc_vals:
    mass_loss_rate =  Q_pc * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    PC_escape.append(mass_loss_rate_atmospheres.value)  
PC_escape = u.Quantity(PC_escape)



# Cusp Escape

Q0_cu_H = 5e24     *( 1/u.s)           
Qmax_cu_H = 5e25   *( 1/u.s) 

# Calculate r_c_E for scaling
nsw = (1e7) * (1 / u.m**3)  
vsw = (604000) * (u.m / u.s)   
r_c_E = ((magnetic_p * Mag_Earth**2) / (8 * np.pi**2 * nsw * vsw**2 * prot_mass))**(1/6)

CU_escape = []
cusp_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    r_c_b = R_MP
    if R_MP > rb_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_cu = min(Q0_cu_H * (r_c_b / r_c_E)**2, Qmax_cu_H) * (Omega_pc / Omega_pc_E) * (rb_exo / r_exo_E)**2
    cusp_vals.append(Q_cu)

for Q_cu in cusp_vals:
    mass_loss_rate =  Q_cu * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    CU_escape.append(mass_loss_rate_atmospheres.value)

CU_escape = u.Quantity(CU_escape)


# Peter Driscoll's Magnetic-Limited Escape 

MPU_escape = []
magnetic_limited_vals = []

time = 10  * (u.s)
mass_water = 1e21 * (u.kg)
sigma_coll = (1 * 10**(-17))  * (u.m**2)
energy_eff = (1 / (10.6e44))

for M in magnetic_moments:
    r_mp = (((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6))  
    h_exo_b = (k_B * T_exo_H * rb_exo**2) / (G * M_B * mass)
    n_exo_b = (1 / (h_exo_b * sigma_coll))  #.to(1 / u.m**3)
    n_L_b = n_exo_b * np.exp((-rb_exo / h_exo_b) * (1 - (rb_exo / r_mp)))
    surface_a_b = (4 * np.pi * r_mp**2)
    f_lm_b = (((energy_eff * h_exo_b * surface_a_b * (n_L_b - nb_sw)) / time) * mass_water) #.decompose()
    magnetic_limited_vals.append(f_lm_b)

for f_lm_b in magnetic_limited_vals:
    mass_loss_rate =  f_lm_b * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    MPU_escape.append(mass_loss_rate_atmospheres.value)

MPU_escape =u.Quantity(MPU_escape)



# Calculate energy-limited escape rate


a = (20.843 * 84e6) * (u.m)     # Used ratio from Eric Agol paper ( a/R*)
M_star = 0.09*M_sun
epsilon = 0.1  # heating efficiency

# Stellar XUV Luminosity
L_bol = 3.828e26 * u.W       # solar luminosity
L_xuv = 3.4e-4 * L_bol       
F_xuv = L_xuv / (4 * np.pi * a**2)                      # flux at planet

# Its labeled as the mass density of the plnet. I used the formula to calculate density: mass/V  
rho = M_B / ((4/3) * np.pi * R_B**3)                    

# Roche lobe radius (Becker+2020 uses 2 in denominator!)
r_roche = a * (M_B / (2 * M_star))**(1/3)

# Tidal enhancement factor K
K = 1 - (3/2) * (R_B / r_roche) + (1/2) * (R_B / r_roche)**3

# Energy-limited escape rate (mass loss) in kg/s
mdot = epsilon * (R_B / R_B)**2 * (3 * F_xuv) / (4 * G * rho * K)
mdot = mdot.to(u.kg / u.s)

seconds_per_gyr = (1e9 * 365.25 * 24 * 3600) * u.s
mdot_earth_per_gyr = (mdot * seconds_per_gyr / M_earth)


mdot_array = u.Quantity(np.full(magnetic_moments.shape, mdot_earth_per_gyr.value), unit=mdot_earth_per_gyr.unit)



# Total Escape  
total_escape = PU_escape.value + CU_escape.value + PC_escape.value + CU_escape.value + MPU_escape.value




plt.figure(figsize=(10, 7))

#plt.loglog(magnetic_m_values_earth_units, mdot_array, label='Energy Limited Escape', color='red', linewidth=3)
plt.loglog(magnetic_m_values_earth_units,  MPU_escape, label = 'Driscoll Ion Pickup Escape', color = 'yellow')
#plt.loglog(magnetic_m_values_earth_units, PU_escape, '--', label='Pickup H Escape', color='orange')
#plt.loglog(magnetic_m_values_earth_units, CF_escape, '-', label='Cross-field H Escape', color='blue')
#plt.loglog(magnetic_m_values_earth_units, PC_escape, label='Polar Cap H Escape', color='purple')
#plt.loglog(magnetic_m_values_earth_units, CU_escape, '--', label='Cusp H Escape', color='hotpink')
#plt.loglog(magnetic_m_values_earth_units, total_escape, linewidth=2.5, color='black', label='Total H Escape')


plt.xlabel('Magnetic Dipole Moment [A·m²]')
plt.ylabel('Escape Rate [s⁻¹]')
plt.title('TRAPPIST-1b Total Hydrogen Escape vs Magnetic Moment')
plt.legend()
#plt.ylim(1e-35, 1e5)
#plt.xlim(1e-40, 1e10)


plt.show()


# In[14]:


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
hb_H = (k_B * T_exo_H * rb_exo**2) / (G * M_B * m_H)

PU_escape = []
pickup_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    if R_MP <= rb_exo:
        Q_pu = Q0_pu_H
    else:
        num = 2*hb_H**3 + 2*hb_H**2*R_MP + hb_H*R_MP**2
        den = 2*hb_H**3 + 2*hb_H**2*rb_exo + hb_H*rb_exo**2
        exp_term = np.exp((rb_exo - R_MP) / hb_H)
        Q_pu = Q0_pu_H * (num / den) * exp_term
    pickup_vals.append(Q_pu)

for Q_pu in pickup_vals:
    mass_loss_rate =  Q_pu * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    PU_escape.append(mass_loss_rate_atmospheres.value)  
PU_escape = u.Quantity(PU_escape)



# Cross-Field Ion Loss

Q0_cf_H = 7.7e25   * (1/u.s)

CF_escape = []
cf_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    if R_MP >= rb_IMB:
        temp = 1 - (rb_exo / R_MP)
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

Q0_pc_H = 7.8e25     * (1/u.s)

PC_escape = []
pc_vals = []
for M in magnetic_moments:
    R_MP = (((magnetic_p * formfact_o**2 * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6))
    if R_MP > rb_IMB:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp)) if temp >= 0 else 0
    else:
        Omega_pc = 0
    Q_pc = Q0_pc_H * (Omega_pc / Omega_pc_E) * ((rb_exo / r_exo_E)**2)
    pc_vals.append(Q_pc)


for Q_pc in pc_vals:
    mass_loss_rate =  Q_pc * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    PC_escape.append(mass_loss_rate_atmospheres.value)  
PC_escape = u.Quantity(PC_escape)



# Cusp Escape

Q0_cu_H = 5e24     *( 1/u.s)           
Qmax_cu_H = 5e25   *( 1/u.s) 

# Calculate r_c_E for scaling
nsw = (1e7) * (1 / u.m**3)  
vsw = (604000) * (u.m / u.s)   
r_c_E = ((magnetic_p * Mag_Earth**2) / (8 * np.pi**2 * nsw * vsw**2 * prot_mass))**(1/6)

b_Omega_pc_values = []
CU_escape = []
cusp_vals = []
for M in magnetic_moments:
    R_MP = ((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6)
    r_c_b = R_MP
    
    if rb_IMB <= R_MP:
        temp = 1 - (r_exo_E / R_MP)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp))
    else:
        Omega_pc = 0
    b_Omega_pc_values.append(Omega_pc)

    
    Q_cu = min(Q0_cu_H * (r_c_b / r_c_E)**2, Qmax_cu_H) * (Omega_pc / Omega_pc_E) * (rb_exo / r_exo_E)**2
    cusp_vals.append(Q_cu)

#for Q_cu in cusp_vals:
    mass_loss_rate =  Q_cu * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    CU_escape.append(mass_loss_rate_atmospheres.value)

CU_escape = u.Quantity(CU_escape)


# Peter Driscoll's Magnetic-Limited Escape 

MPU_escape = []
magnetic_limited_vals = []

time = 10  * (u.s)
mass_water = 1e21 * (u.kg)
sigma_coll = (1 * 10**(-17))  * (u.m**2)
energy_eff = (1 / (10.6e44))

for M in magnetic_moments:
    r_mp = (((magnetic_p * M**2) / (8 * np.pi**2 * nb_sw * vb_sw**2 * prot_mass))**(1/6))  
    h_exo_b = (k_B * T_exo_H * rb_exo**2) / (G * M_B * mass)
    n_exo_b = (1 / (h_exo_b * sigma_coll))  #.to(1 / u.m**3)
    n_L_b = n_exo_b * np.exp((-rb_exo / h_exo_b) * (1 - (rb_exo / r_mp)))
    surface_a_b = (4 * np.pi * r_mp**2)
    f_lm_b = (((energy_eff * h_exo_b * surface_a_b * (n_L_b - nb_sw)) / time) * mass_water) #.decompose()
    magnetic_limited_vals.append(f_lm_b)

for f_lm_b in magnetic_limited_vals:
    mass_loss_rate =  f_lm_b * mass          # Multiply by mass of hydrogen to get units in kg/s
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    MPU_escape.append(mass_loss_rate_atmospheres.value)

MPU_escape =u.Quantity(MPU_escape)



# Calculate energy-limited escape rate


a = (20.843 * 84e6) * (u.m)     # Used ratio from Eric Agol paper ( a/R*)
M_star = 0.09*M_sun
epsilon = 0.1  # heating efficiency

# Stellar XUV Luminosity
L_bol = 3.828e26 * u.W       # solar luminosity
L_xuv = 3.4e-4 * L_bol       
F_xuv = L_xuv / (4 * np.pi * a**2)                      # flux at planet

# Its labeled as the mass density of the plnet. I used the formula to calculate density: mass/V  
rho = M_B / ((4/3) * np.pi * R_B**3)                    

# Roche lobe radius (Becker+2020 uses 2 in denominator!)
r_roche = a * (M_B / (2 * M_star))**(1/3)

# Tidal enhancement factor K
K = 1 - (3/2) * (R_B / r_roche) + (1/2) * (R_B / r_roche)**3

# Energy-limited escape rate (mass loss) in kg/s
mdot = epsilon * (R_B / R_B)**2 * (3 * F_xuv) / (4 * G * rho * K)
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
plt.ylabel('Escape Rate [s⁻¹]')
plt.title('TRAPPIST-1b Total Hydrogen Escape vs Magnetic Moment')
plt.legend()
plt.ylim(1e-35, 1e5)
plt.xlim(1e-4, 1e2)

#plt.ylim(1e-5, 1e0)
#plt.xlim(1e2, 1e5)

plt.show()


# In[ ]:


# Check

