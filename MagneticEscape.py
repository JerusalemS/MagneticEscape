import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import k_B, G


# Same constants for all
mass_earth = 5.972e24 * u.kg  
earth_atmosphere_in_kg = (1e-6 * mass_earth)  
seconds_per_gyr = 1e9 * 365.25 * 24 * 3600

# Constants for Ion-pickup
rE_exo = 6871000 * u.m
prot_mass = (1.67e-27) * (u.kg)
magnetic_p = (4 * np.pi * 10**(-7)) * (u.H / u.m)  
mass_h = (3.35 * 10**(-24)) * u.g  
time = 10 * (u.s)  
mass_water = (1e21 * (u.kg)) 


# Constants for second plot
mass_oxygen = (2.66e-26) * u.kg

# Energy Limited Escape
K = 0.55  
m_star = 1.77e29 * u.kg  

# Lists for results of all plots
emass_loss_rates = []
escape_rates_results = []
Omega_pc_values = []
R_mp_values = []

# Parameter for the first plot (Ion-pickup Escape)
emagnetic_m_values = np.logspace(18, 27, 200) * (u.A * u.m**2)  
t_exo = 1107 * u.K  
sigma_coll = (1e-17) * (u.m**2) 

n_sw = (1e7) * (1/ u.m**3)  
v_sw = 604000 * (u.m / u.s)   
sw_pressure_e = (prot_mass * n_sw * (v_sw**2))
r_exo = 6572900 * u.m

# Parameter for the second plot (Polar Cap Escape)
Q0_pce_O = 2.83e25  
Omega_e = 2.65
r_imb = 6666800 * u.m  
magnetic_moment_e = 8e22 * (u.A * u.m**2)

# Magnetic moments for the second plot
magnetic_moments = np.logspace(17, 27, 200) * (u.A * u.m**2)

# Energy limited escape
I_XUV_values = 0.85 *(u.W/u.m**2)  
r_pl_e = 6067886.04 * (u.m) 
m_pl_e = 3.9964624e24 * u.kg  
m_star = 1.77e29 * u.kg  
magnetic_m_values = np.logspace(18, 22, 200) * (u.A * u.m**2)  

# Calculate energy-limited escape rate
Gamma = (r_pl_e * (r_pl_e**2) * I_XUV_values) / (m_pl_e * G * K)  
Gamma_atmospheres = ((Gamma / earth_atmosphere_in_kg) * seconds_per_gyr).decompose() 

# Create an array of constant escape rates
Gamma_atmospheres_array = np.full_like(magnetic_m_values.value, Gamma_atmospheres.value)

# First Plot Calculations
for magnetic_m in emagnetic_m_values:
    r_mp_e = ((magnetic_p * magnetic_m**2) / (8 * np.pi**2 * (sw_pressure_e)))**(1/6)
    h_exo_e = (k_B * t_exo / (mass_h * 7 * (u.m / u.s**2))).decompose()  
    n_exo_e = (1 / (h_exo_e * sigma_coll)).to(1 / u.m**3)
    n_L_e = n_exo_e * np.exp((-r_exo / h_exo_e) * (1 - (r_exo / r_mp_e)))
    energy_eff = 1 / (10.6 * 10**44)
    surface_a = (4 * np.pi * r_mp_e**2)
    f_lm_e = (((energy_eff * h_exo_e * surface_a * (n_L_e - n_sw)) / time) * mass_water).decompose()
    
    emass_loss_rate_earth_atmospheres = (f_lm_e / earth_atmosphere_in_kg) * seconds_per_gyr
    emass_loss_rates.append(emass_loss_rate_earth_atmospheres.value)

# Second Plot Calculations
for magnetic_moment in magnetic_moments:
    R_mp = (((magnetic_p * (magnetic_moment**2)) / 
              (8 * (np.pi**2) * n_sw * (v_sw**2) * prot_mass))**(1/6)).decompose()
    R_mp_values.append(R_mp.value)

    if r_imb <= R_mp:
        temp_value = 1 - (rE_exo / R_mp)
        Omega_pc = 4 * np.pi * (1 - np.sqrt(temp_value)) if temp_value >= 0 else 0
    else:
        Omega_pc = 0  

    Omega_pc_values.append(Omega_pc)
    escape_rate = Q0_pce_O * (Omega_pc / Omega_e) * ((r_exo / rE_exo)**2)
    mass_loss_rate = escape_rate * mass_oxygen
    mass_loss_rate_atmospheres = (mass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    escape_rates_results.append(mass_loss_rate_atmospheres.value)  

# Convert magnetic moments to Earth units for plotting
emagnetic_m_values_earth_units = emagnetic_m_values / (8e22 * (u.A * u.m**2))

emagnetic_m_values_plot = emagnetic_m_values_earth_units 
emass_loss_rates_plot = np.array(emass_loss_rates) 


magnetic_moments_earth_units = magnetic_moments / magnetic_moment_e  
    

# Plotting
plt.figure(figsize=(12, 8))

# First plot (Mass Loss Rate for TRAPPIST-1e)
plt.loglog(magnetic_moments_earth_units, Gamma_atmospheres_array, label='Energy Limited Escape', color='red', linewidth=3)
plt.loglog(emagnetic_m_values_plot, emass_loss_rates_plot, label='Ion Pickup Escape', linewidth=3)
plt.loglog(magnetic_moments_earth_units, escape_rates_results, label='Polar Cap Escape', color='orange', linewidth=3)
plt.xlabel('Magnetic Dipole Moment (Earth Units)',fontsize=24)
plt.ylabel('Escape Rate (Earth Atmospheres/Gyr)', fontsize=24)
plt.title('TRAPPIST-1 e', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
plt.ylim(1e-4, 1e4)
plt.xlim(1e-3, 1e2)
plt.legend(fontsize=16)
plt.tight_layout()  
plt.show()

# Parameters for the first plot

gmagnetic_m_values = np.logspace(18, 27, 200) * (u.A * u.m**2)  
t_exo = 1107 * u.K  
sigma_coll = 1e-17 * (u.m**2) 
n_sw_g = (1e7) * (1 / u.m**3)  
R_tg = 7200916.7 * (u.m)
v_sw_g = (637000) * (u.m / u.s)   
r_exo_g = 396 * (u.km) + R_tg
sw_pressure_g = (prot_mass*n_sw_g*(v_sw_g**2))


# Parameters for the second plot (Oxygen Escape Rate)
R_exo_g = 396000 * u.m + R_tg
Q0_pcg_O = 2.71e25  
r_imb = 6666800 * u.m  
magnetic_moment_e = 8e22 * (u.A * u.m**2)
Omega_pc_E = 0.67
rE_exo = 7071000 * u.m 


# Magnetic moments for the second plot
magnetic_moments = np.logspace(17, 27, 200) * (u.A * u.m**2)

# Lists for results
gmass_loss_rates = []
escape_rates_gresults = []
Omega_pc_gvalues = []
R_mp_gvalues = []


# Energy limited escape
r_pl_g = 7200916.7 * u.m
m_pl_g = 7.89e24 * u.kg  
m_star = 1.77e29 * u.kg  
I_XUV_values_g = 0.32  


# Calculate energy-limited escape rate
Gamma_g = (r_pl_g * (r_pl_g**2) * I_XUV_values_g) / (m_pl_g * G * K)  
Gamma_atmospheres_g = (Gamma_g / earth_atmosphere_in_kg) * seconds_per_gyr  

# Create an array of constant escape rates
Gamma_atmospheres_garray = np.full_like(gmagnetic_m_values.value, Gamma_atmospheres_g.value)


# First Plot Calculations
for magnetic_m in gmagnetic_m_values:
    r_mp_g = ((magnetic_p * magnetic_m**2) / (8 * np.pi**2 * sw_pressure_g))**(1/6)
    h_exo_g = (k_B * t_exo / (mass_h * (9.12 * (u.m / u.s**2)))).decompose()  
    n_exo_g = (1 / (h_exo_g * sigma_coll)).to(1 / u.m**3)
    n_L_g = n_exo_g * np.exp((-r_exo_g / h_exo_g) * (1 - (r_exo_g / r_mp_g)))
    energy_eff = 1 / (10.6 * 10**44)
    surface_a_g = (4 * np.pi * r_mp_g**2)
    f_lm_g = (((energy_eff * h_exo_g * surface_a_g * (n_L_g - n_sw_g)) / time) * mass_water).decompose()
    
    gmass_loss_rate_earth_atmospheres = (f_lm_g / earth_atmosphere_in_kg)*seconds_per_gyr  # Convert to Gyr
    gmass_loss_rates.append(gmass_loss_rate_earth_atmospheres.value)


# Second Plot Calculations
for magnetic_moment in magnetic_moments:
    R_mp_g = (((magnetic_p * (magnetic_moment**2)) / 
              (8 * (np.pi**2) * n_sw_g * (v_sw_g**2) * prot_mass))**(1/6)).decompose()

    R_mp_gvalues.append(R_mp_g.value)

    if r_imb <= R_mp_g:
        temp_value = 1 - (rE_exo / R_mp_g)
        Omega_pc_g = 4 * np.pi * (1 - np.sqrt(temp_value)) if temp_value >= 0 else 0
    else:
        Omega_pc_g = 0  

    Omega_pc_gvalues.append(Omega_pc_g)
    escape_rate_g = Q0_pcg_O * (Omega_pc_g / Omega_pc_E) * ((R_exo_g / rE_exo)**2)
    
    gmass_loss_rate = escape_rate_g * mass_oxygen
    gmass_loss_rate_atmospheres = (gmass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    escape_rates_gresults.append(gmass_loss_rate_atmospheres.value)  

# Convert magnetic moments to Earth units for plotting
gmagnetic_m_values_earth_units = gmagnetic_m_values / (8e22 * (u.A * u.m**2))
gmagnetic_moments_earth_units = magnetic_moments / magnetic_moment_e  

#1 plot
gmass_loss_rates_plot = np.array(gmass_loss_rates) 


# Plotting
plt.figure(figsize=(12, 8))

# First plot (Mass Loss Rate for TRAPPIST-1e)
plt.loglog(gmagnetic_moments_earth_units, Gamma_atmospheres_garray, label='Energy Limited Escape', color='red', linewidth=3)
plt.loglog(gmagnetic_m_values_earth_units, gmass_loss_rates_plot, label='Ion Pickup Escape', linewidth=3)
plt.loglog(gmagnetic_moments_earth_units, escape_rates_gresults, label='Polar Cap Escape', color='orange', linewidth=3)
plt.xlabel('Magnetic Dipole Moment (Earth Units)',fontsize=24)
plt.ylabel('Escape Rate (Earth Atmospheres/Gyr)', fontsize=24)
plt.title('TRAPPIST-1g', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
plt.ylim(1e-3, 1e4)
plt.xlim(1e-3, 1e0)
plt.legend(fontsize=16)
plt.show()

# Parameters for the first plot( Ion-Pickup Escape)
magnetic_m_values = np.logspace(18, 27, 200) * (u.A * u.m**2)  
t_exo = 1107 * u.K  
sigma_coll = 1e-17 * u.m**2 
R_tf = 6665153.2 * u.m
r_exo_f = 396 * (u.km) + R_tf  


# Parameters for the second plot (Oxygen Escape Rate)
R_exo_f = 396000 * u.m + R_tf
n_sw_f = (1e7) * (1 / u.m**3)  
v_sw_f = (624000) * (u.m / u.s)   
magnetic_moment_f = (8e22) * (u.A * u.m**2)
Q0_pcf_O = 3.37e25  
sw_pressure_f = (prot_mass*n_sw_f*(v_sw_f**2))
r_imb = 6666800 * u.m  
magnetic_moment_e = 8e22 * (u.A * u.m**2)
Omega_pc_E = 0.67

# Magnetic moments for the second plot
magnetic_moments = np.logspace(17, 27, 200) * (u.A * u.m**2)

# Lists for results
fmass_loss_rates = []
escape_rates_fresults = []
Omega_pc_fvalues = []
R_mp_fvalues = []


# Energy limited escape
r_pl_f = 6665153.2 * u.m
m_pl_f = 6.2e24 * u.kg  
I_XUV_values_f = 0.48


# Calculate energy-limited escape rate
Gamma_f = (r_pl_f * (r_pl_f**2) * I_XUV_values_f) / (m_pl_f * G * K)  
Gamma_atmospheres_f = (Gamma_f / earth_atmosphere_in_kg) * seconds_per_gyr  

# Create an array of constant escape rates
Gamma_atmospheres_farray = np.full_like(magnetic_m_values.value, Gamma_atmospheres_f.value)


# First Plot Calculations
for magnetic_mf in magnetic_m_values:
    r_mp_f = (((magnetic_p * magnetic_mf**2) / (8 * np.pi**2 * sw_pressure_f))**(1/6)).decompose()
    h_exo_f = (k_B * t_exo / (mass_h * 8.3 * (u.m / u.s**2))).decompose()  
    n_exo_f = (1 / (h_exo_f * sigma_coll)).to(1 / u.m**3)
    n_L_f = n_exo_f * np.exp((-r_exo_f / h_exo_f) * (1 - (r_exo_f / r_mp_f)))
    energy_eff = (1 / (10.6 * 10**44))
    surface_a_f = (4 * np.pi * r_mp_f**2)
    f_lm_f = (((energy_eff * h_exo_f * surface_a_f * (n_L_f - n_sw_f)) / time) * mass_water).decompose()
    
    fmass_loss_rate_earth_atmospheres = (f_lm_f / earth_atmosphere_in_kg)*seconds_per_gyr  # Convert to Gyr
    fmass_loss_rates.append(fmass_loss_rate_earth_atmospheres.value)


# Second Plot Calculations
for magnetic_moment in magnetic_moments:
    R_mp_f = (((magnetic_p * (magnetic_moment**2)) / 
                (8 * (np.pi**2) * n_sw_f * (v_sw_f**2) * prot_mass))**(1/6)).decompose()

    R_mp_fvalues.append(R_mp_f.value)

    if r_imb <= R_mp_f:
        temp_value = 1 - (rE_exo / R_mp_f)  # Use R_mp_f instead of R_mp
        Omega_pc_f = 4 * np.pi * (1 - np.sqrt(temp_value)) if temp_value >= 0 else 0
    else:
        Omega_pc_f = 0  

    Omega_pc_fvalues.append(Omega_pc_f)

    escape_rate_f = Q0_pcf_O * (Omega_pc_f / Omega_pc_E) * ((R_exo_f / rE_exo)**2)

    fmass_loss_rate = escape_rate_f * mass_oxygen
    fmass_loss_rate_atmospheres = (fmass_loss_rate / earth_atmosphere_in_kg) * seconds_per_gyr  
    escape_rates_fresults.append(fmass_loss_rate_atmospheres.value)


# Convert magnetic moments to Earth units for plotting
fmagnetic_m_values_earth_units = magnetic_m_values / (8e22 * (u.A * u.m**2))
fmagnetic_moments_earth_units = magnetic_moments / magnetic_moment_e  

#1 plot
fmass_loss_rates_plot = np.array(fmass_loss_rates) 
# Plotting
plt.figure(figsize=(12, 8))

# First plot (Mass Loss Rate for TRAPPIST-1e)
plt.loglog(fmagnetic_moments_earth_units, Gamma_atmospheres_farray, label='Energy Limited Escape', color='red', linewidth=3)
plt.loglog(fmagnetic_m_values_earth_units, fmass_loss_rates_plot, label='Ion Pickup Escape', linewidth=3)
plt.loglog(fmagnetic_moments_earth_units, escape_rates_fresults, label='Polar Cap Escape', color='orange', linewidth=3)
plt.xlabel('Magnetic Dipole Moment (Earth Units)',fontsize=24)
plt.ylabel('Escape Rate (Earth Atmospheres/Gyr)', fontsize=24)
plt.title('TRAPPIST-1f', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
plt.ylim(1e-4, 1e4)
plt.xlim(1e-3, 1e0)
plt.legend(fontsize=16)
plt.show()
# checking for commit command
