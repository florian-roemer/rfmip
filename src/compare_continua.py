# %%
import pyarts
import numpy as np
from experiment_setup import read_exp_setup
from data_visualisation import convert_units
import matplotlib.pyplot as plt
import os
import scipy.integrate
import xarray as xr
import typhon

def mov_avg(array, N, axis=0, win_type=None):
    # calculate moving average of a numpy array
    import pandas as pd

    if len(array.shape) == 1:
        if win_type == None:
            mov_avg = pd.Series(array).rolling(
                window=N, center=True, win_type=win_type,
                min_periods=int(N/2)).mean().iloc[:].values
        else:
            mov_avg = pd.Series(array).rolling(
                window=N, center=True, win_type=win_type,
                min_periods=int(N/2)).mean(std=N/2).iloc[:].values

    elif len(array.shape) == 2:
        if win_type == None:
            mov_avg = pd.DataFrame(array).rolling(
                window=N, center=True, win_type=win_type,
                min_periods=int(N/2), axis=axis).mean(std=N/2).iloc[:].values
        else:
            mov_avg = pd.DataFrame(array).rolling(
                window=N, center=True, win_type=win_type,
                min_periods=int(N/2), axis=axis).mean(std=N/2).iloc[:].values
                
    return mov_avg


def spectral_plot(ax):
    ax.hlines(0, 1, 2500, color='k', 
          linewidth=ax.spines['bottom'].get_linewidth())
    ax.set_xlim([1, 2500])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylabel(r"spectral irradiance / mW$\,$m$^{-2}\,$cm")
    ax.set_xlabel("wavenumber / cm-1")
    
    return ax


def iwv_plot(ax):
    ax.hlines(0, 0, 65, color='k', 
          linewidth=ax.spines['bottom'].get_linewidth())
    ax.set_xlim([0, 65])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylabel(r"Flux / W$\,$m$^{-2}$", fontsize=14)
    ax.set_xlabel(r"WVC / kg$\,$m$^{-2}$", fontsize=14)
    
    return ax

os.chdir('/Users/froemer/Documents/wv_continuum/rfmip')

exp_setup = read_exp_setup(exp_name='olr', 
                           path='experiment_setups/')
spectral_grid = np.linspace(
    exp_setup.spectral_grid["min"],
    exp_setup.spectral_grid["max"],
    exp_setup.spectral_grid["n"],
    endpoint=True)

atm = xr.open_dataset(
    f"/Users/froemer/Documents/rte-rrtmgp/examples/rfmip-clear-sky/"
    "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc")

weight = atm.profile_weight
iwv = typhon.physics.integrate_water_vapor(
    atm.water_vapor[0, :, ::-1], atm.pres_layer[:, ::-1], axis=1)

# dimensions: site, wavelength, pressure, 1, 1, down-/upward
# OLR: [:,:, -1, 0, 0, 1]
# surface downwelling: [:, :, 0, 0, 0, 0]
data = np.zeros((4, 100, 1000))
for i, cont in enumerate(['True', 'self', 'foreign', 'False']):
    data[i] = data_on = np.array(pyarts.xml.load(
    f"{exp_setup.rfmip_path}output/{exp_setup.name}/"
    f"continua_{cont}/spectral_irradiance.xml"))[:,:, -1, 0, 0, 1]  
 
wavenumber, irradiance_on = convert_units(
    exp_setup=exp_setup, spectral_grid=spectral_grid, irradiance=data[0])
wavenumber, irradiance_self = convert_units(
    exp_setup=exp_setup, spectral_grid=spectral_grid, irradiance=data[1])
wavenumber, irradiance_foreign = convert_units(
    exp_setup=exp_setup, spectral_grid=spectral_grid, irradiance=data[2])
wavenumber, irradiance_off= convert_units(
    exp_setup=exp_setup, spectral_grid=spectral_grid, irradiance=data[3])

olr_on = scipy.integrate.trapz(irradiance_on, wavenumber)
olr_self = scipy.integrate.trapz(irradiance_self, wavenumber)
olr_foreign = scipy.integrate.trapz(irradiance_foreign, wavenumber)
olr_off = scipy.integrate.trapz(irradiance_off, wavenumber)

# %%
strange = olr_on > olr_off
inversion = atm.temp_layer[0, :, 36:].max(axis=1) > atm.surface_temperature[0]
select = abs(atm.lat) <= 90
# select = (atm.surface_temperature[0] > 284) & \
#     (atm.surface_temperature[0] < 302) \
#         & ~inversion
# select = ~inversion

olr_on_avg = np.average(olr_on[select], weights=weight[select])
olr_self_avg = np.average(olr_self[select], weights=weight[select])
olr_foreign_avg = np.average(olr_foreign[select], weights=weight[select])
olr_off_avg = np.average(olr_off[select], weights=weight[select])

print(f'effect total continuum: {olr_on_avg-olr_off_avg} W m-2')
print(f'effect self continuum: {olr_self_avg-olr_off_avg} W m-2')
print(f'effect foreign continuum: {olr_foreign_avg-olr_off_avg} W m-2')

# %%
select = abs(atm.lat) <= 90
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax = spectral_plot(ax)
ax.set_title('change in OLR caused by WV continuum',
             fontsize=16)

colors = plt.get_cmap('coolwarm')
quantity = atm.surface_temperature[0]
# quantity = iwv
colorlist = colors((quantity - quantity.min())/
                   (quantity.max() - quantity.min()))

for i in range(len(colorlist)):
    if select[i]:
        ax.plot(wavenumber,
                mov_avg(irradiance_on[i].T - irradiance_off[i].T, 20)*1e3,
                color=colorlist[i])
ax.plot(wavenumber,
        np.average(
            mov_avg(irradiance_on[select].T - irradiance_off[select].T, 20)*1e3, 
            axis=1, weights=weight[select]), color='k', linewidth=3)
# cbar = plt.colorbar(colors)
# ax.set_ylim([-15,5])

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax = spectral_plot(ax)
ax.set_title('global mean change in OLR caused by WV continuum',
             fontsize=16)
ax.plot(wavenumber, 
        mov_avg(
            np.average(irradiance_on - irradiance_off, axis=0, weights=weight),
            20)*1e3, 
        label='total')
ax.plot(wavenumber, 
        mov_avg(
            np.average(irradiance_self - irradiance_off, axis=0, weights=weight),
            20)*1e3, 
        label='self')
ax.plot(wavenumber, 
        mov_avg(
            np.average(irradiance_foreign - irradiance_off, axis=0, weights=weight),
            20)*1e3, 
        label='foreign')
ax.legend(fontsize=14)
# %%
# fig, ax = plt.subplots(1, 1, figsize=(3, 5))

# ax.set_title('temperature', fontsize=18)
# ax.plot(atm.temp_layer[0, :].T,
#         atm.pres_layer[:].T/1e2, 
#         color='grey',
#         alpha=0.5, linewidth=0.5)
# ax.plot(atm.temp_layer[0, strange].T,
#         atm.pres_layer[strange].T/1e2, 
#         color='green', alpha=0.5, linewidth=0.5)
# ax.scatter(atm.surface_temperature[0], 
#            atm.pres_layer[:, -1].T/1e2,
#            s=10, color='grey')
# ax.scatter(atm.surface_temperature[0, strange], 
#            atm.pres_layer[strange, -1].T/1e2,
#            s=10, color='green')

# ax.set_ylim([atm.pres_layer.max()/1e2, 
#              atm.pres_layer.min()/1e2])


# %%
# contingency table
inversions_strange = len(np.where(strange[inversion])[0])
inversions_normal = len(np.where(~strange[inversion])[0])
not_inversions_strange = len(np.where(strange[~inversion])[0])
not_inversions_normal = len(np.where(~strange[~inversion])[0])

n = inversions_normal + inversions_strange \
    + not_inversions_normal + not_inversions_strange
    
hit_rate = inversions_strange / (inversions_strange + not_inversions_strange)
false_alarm_rate = inversions_normal / (inversions_normal + not_inversions_normal)
peirce_skill_score = hit_rate - false_alarm_rate

print(np.round(peirce_skill_score, 2))
# %%
# fig, ax = plt.subplots(1, 2, figsize=(7, 7))
# ax[0].set_title('temperature', fontsize=16)
# ax[0].plot(atm.temp_layer[0, 20, :],
#          atm.pres_layer[20, :]/100)
# ax[0].set_ylim(1000, 100)
# ax[0].scatter(atm.surface_temperature[0, 20], atm.pres_layer[20, -1]/100)

# ax[1].set_title('H2O VMR', fontsize=16)
# ax[1].plot(atm.water_vapor[0, 20, :],
#          atm.pres_layer[20, :]/100)
# ax[1].set_ylim(1000, 100)

# %% reproduce Paynter & Ramaswamy (2012)
# select = ~inversion
select = atm.surface_temperature[0] > 0
print(f"{len(atm.lat[select])}/100 sites selected")

fig, ax = plt.subplots(2, 2, figsize=(20, 13))
colors = plt.get_cmap('inferno')
quantity = atm.surface_temperature[0]
colorlist = colors(((quantity - 284)/(302 - 284)))

ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1] = iwv_plot(ax[0, 0]),\
    iwv_plot(ax[0, 1]), iwv_plot(ax[1, 0]), iwv_plot(ax[1, 1])

ax[0, 0].set_ylim([olr_off[select].min(), olr_off[select].max()])

ax[0, 0].set_title('OLR with no Continuum', fontsize=18)
ax[0, 0].scatter(iwv[select], olr_off[select], 
                 color=np.array(colorlist)[select])
ax[0, 1].set_title('Reduction in OLR due to Continuum', fontsize=18)
ax[0, 1].scatter(iwv[select], -(olr_on - olr_off)[select], 
                 color=np.array(colorlist)[select])
ax[1, 0].set_title('Reduction in OLR due to Self Continuum', fontsize=18)
ax[1, 0].scatter(iwv[select], -(olr_self - olr_off)[select], 
                 color=np.array(colorlist)[select])
ax[1, 1].set_title('Reduction in OLR due to Foreign Continuum', fontsize=18)
ax[1, 1].scatter(iwv[select], -(olr_foreign - olr_off)[select], 
                 color=np.array(colorlist)[select])

# %%
select = abs(atm.lat) <= 90
# select = ~inversion
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
ax[0] = spectral_plot(ax[0])
ax[0].set_title('change in OLR caused by self continuum',
             fontsize=16)
ax[0].set_title('change in OLR caused by foreign continuum',
             fontsize=16)

# colors = plt.get_cmap('gist_rainbow_r')
colors = plt.cm.get_cmap('gist_rainbow_r')
# quantity = atm.surface_temperature[0]
quantity = iwv

liste = quantity
colorlist = colors((quantity - quantity.min())/
                   (quantity.max() - quantity.min())+0.1)

for i in range(len(colorlist)):
    if select[i]:
        a = ax[0].scatter(wavenumber,
                -mov_avg(irradiance_self[i].T - irradiance_off[i].T, 20)*1e3,
                c=np.tile(quantity[i], len(wavenumber)), vmin=0, vmax=60, s=1, cmap=colors)
        a = ax[1].scatter(wavenumber,
                -mov_avg(irradiance_foreign[i].T - irradiance_off[i].T, 20)*1e3,
                c=np.tile(quantity[i], len(wavenumber)), vmin=0, vmax=60, s=1, cmap=colors)

plt.colorbar(a)

# %%
