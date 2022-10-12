# %%
import pyarts
import numpy as np
from experiment_setup import read_exp_setup
from data_visualisation import convert_units
import matplotlib.pyplot as plt
import os
import scipy.integrate
import xarray as xr

def mov_avg(array, N, axis=0, win_type=None):
    # calculate moving average of a numpy array
    import pandas as pd

    if len(array.shape) == 1:
        if win_type == None:
            mov_avg = pd.Series(array).rolling(window=N, center=True, win_type=win_type,
                                            min_periods=int(N/2)).mean().iloc[:]\
                                            .values
        else:
            mov_avg = pd.Series(array).rolling(window=N, center=True, win_type=win_type,
                                            min_periods=int(N/2)).mean(std=N/2).iloc[:]\
                                            .values

    elif len(array.shape) == 2:
        if win_type == None:
            mov_avg = pd.DataFrame(array).rolling(window=N, center=True, win_type=win_type,
                min_periods=int(N/2), axis=axis).mean(std=N/2).iloc[:].values
        else:
            mov_avg = pd.DataFrame(array).rolling(window=N, center=True, win_type=win_type,
                min_periods=int(N/2), axis=axis).mean(std=N/2).iloc[:].values
                
    return mov_avg

os.chdir('/Users/froemer/Documents/wv_continuum/rfmip')

exp_setup = read_exp_setup(exp_name='olr', path='/Users/froemer/Documents/wv_continuum/rfmip/experiment_setups/')
spectral_grid = np.linspace(
    exp_setup.spectral_grid["min"],
    exp_setup.spectral_grid["max"],
    exp_setup.spectral_grid["n"],
    endpoint=True,
)

atm = xr.open_dataset(
    f"/Users/froemer/Documents/rte-rrtmgp/examples/rfmip-clear-sky/"
    "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
)
weight = atm.profile_weight


# OLR: [:,:, -1, 0, 0, 1]
# surface downwelling: [:,:, 0, 0, 0, 0]


# data_on = np.average(np.array(pyarts.xml.load(
#     f"{exp_setup.rfmip_path}output/{exp_setup.name}/continua_True/spectral_irradiance.xml"
# ))[:,:, -1, 0, 0, 1], axis=0, weights=weight)  # site, wavelength, pressure, 1, 1, down-/upward

# data_off = np.average(np.array(pyarts.xml.load(
#     f"{exp_setup.rfmip_path}output/{exp_setup.name}/continua_False/spectral_irradiance.xml"
# ))[:,:, -1, 0, 0, 1], axis=0, weights=weight)   # site, wavelength, pressure, 1, 1, down-/upward

data_on = np.array(pyarts.xml.load(
    f"{exp_setup.rfmip_path}output/{exp_setup.name}/continua_True/spectral_irradiance.xml"
))[:,:, -1, 0, 0, 1]  # site, wavelength, pressure, 1, 1, down-/upward

data_off = np.array(pyarts.xml.load(
    f"{exp_setup.rfmip_path}output/{exp_setup.name}/continua_False/spectral_irradiance.xml"
))[:,:, -1, 0, 0, 1]  # site, wavelength, pressure, 1, 1, down-/upward



spectral_grid_converted_on, irradiance_converted_on = convert_units(
    exp_setup=exp_setup, spectral_grid=spectral_grid, irradiance=data_on)
spectral_grid_converted_off, irradiance_converted_off= convert_units(
    exp_setup=exp_setup, spectral_grid=spectral_grid, irradiance=data_off)

olr_off = scipy.integrate.trapz(irradiance_converted_off, spectral_grid_converted_off)
olr_on = scipy.integrate.trapz(irradiance_converted_on, spectral_grid_converted_on)

# %%
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

select = abs(atm.lat) <= 90
select = olr_on < olr_off


ax.hlines(0, 1, 2500, color='k', 
         linewidth=ax.spines['bottom'].get_linewidth())
ax.set_xlim([1, 2500])
colors = plt.get_cmap('coolwarm_r')
colorlist = colors(abs(atm.lat)/90)
# colorlist = colors(np.linspace(0, 1, 100))

for i in range(len(colorlist)):
    if select[i]:
        ax.plot(
            spectral_grid_converted_on,
            (1e3*mov_avg(irradiance_converted_on[i].T-
                        irradiance_converted_off[i].T, 20)),
            color=colorlist[i]
        )

ax.plot(
    spectral_grid_converted_on,
    np.average((1e3*mov_avg(irradiance_converted_on[select].T-
                irradiance_converted_off[select].T, 20)), 
               axis=1, weights=weight[select]),
    color='k', linewidth=3
)

# ax.plot(spectral_grid_converted_on[:-1],
#         irradiance_converted_on[:-1])
# ax.plot(spectral_grid_converted_off[:-1],
#         irradiance_converted_off[:-1])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_ylabel(r"spectral irradiance / 10$^{-3}\,$W$\,$m$^{-2}\,$cm")
ax.set_xlabel("wavenumber / cm-1")

# %%
lut_on = pyarts.xml.load('lookup_tables/continua_True/olr.xml')
lut_off = pyarts.xml.load('lookup_tables/continua_False/olr.xml')

p_grid = lut_on.p_grid

xsec_on = lut_on.xsec
xsec_off = lut_off.xsec

# xsec_diff = xsec_on - xsec_off

# %%
