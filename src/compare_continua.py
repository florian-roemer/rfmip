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

    ax.set_ylabel(r"spectral irradiance / mW$\,$m$^{-2}\,$cm", fontsize=18)
    ax.set_xlabel("wavenumber / cm-1", fontsize=18)
    
    ax.set_xticks(np.array(ax.get_xticks(), dtype='int'))
    ax.set_yticks(np.array(ax.get_yticks(), dtype='int'))
   
    ax.set_xticklabels(np.array(ax.get_xticks(), dtype='int'), fontsize=16)
    ax.set_yticklabels(np.array(ax.get_yticks(), dtype='int'), fontsize=16)
    
    return ax


def iwv_plot(ax):
    ax.hlines(0, 0, 65, color='k', 
          linewidth=ax.spines['bottom'].get_linewidth())
    ax.set_xlim([0, 65])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylabel(r"Flux / W$\,$m$^{-2}$", fontsize=18)
    ax.set_xlabel(r"WVC / kg$\,$m$^{-2}$", fontsize=18)

    ax.set_xticks(np.array(ax.get_xticks(), dtype='int'))
    ax.set_yticks(np.array(ax.get_yticks(), dtype='int'))
   
    ax.set_xticklabels(np.array(ax.get_xticks(), dtype='int'), fontsize=16)
    ax.set_yticklabels(np.array(ax.get_yticks(), dtype='int'), fontsize=16)
    
    return ax


def heatingrates_from_fluxes(pressure, downward_flux, upward_flux):
    """taken from: https://github.com/atmtools/konrad/blob/main/konrad/radiation/radiation.py
    Calculate heating rates from radiative fluxes.
    Parameters:
        pressure (ndarray): Pressure half-levels [Pa].
        downward_flux (ndarray): Downward radiative flux [W/m^2].
        upward_flux (ndarray): Upward radiative flux [W/m^2].
    Returns:
        ndarray: Radiative heating rate [K/day].
    """
    from typhon import constants
    
    c_p = constants.isobaric_mass_heat_capacity
    g = constants.earth_standard_gravity

    q = g / c_p * np.diff(upward_flux - downward_flux) / np.diff(pressure)
    q *= 3600 * 24

    return q

os.chdir('/Users/froemer/Documents/wv_continuum/rfmip')

exp_setup = read_exp_setup(exp_name='olr', 
                           path='experiment_setups/')
wavenumber = np.linspace(
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

spectral_irradiance = np.zeros((4, 100, 1000, 60, 2))
for i, cont in enumerate(['True', 'self', 'foreign', 'False']):
    spectral_irradiance[i] = xr.DataArray(pyarts.xml.load(
    f"{exp_setup.rfmip_path}output/{exp_setup.name}/"
    f"continua_{cont}/spectral_irradiance.xml"))[:,:, :, 0, 0, :]  

# dimensions: continuum config, site, wavelength, pressure, down-/upward
_, spectral_irradiance = convert_units(
    exp_setup=exp_setup, spectral_grid=wavenumber, irradiance=spectral_irradiance)

# dimensions: continuum config, site, pressure, down-/upward
irradiance = np.trapz(y=spectral_irradiance, x=wavenumber, axis=2)

toa_up_spectral = spectral_irradiance[:, :, :, -1, 1]
sfc_down_spectral = spectral_irradiance[:, :, :, 0, 0]
 
olr = irradiance[:, :, -1, 1]
sdr = irradiance[:, :, 0, 0]

heating_rates = heatingrates_from_fluxes(atm.pres_layer, 
                                         -irradiance[:, :, ::-1, 0],
                                         irradiance[:, :, ::-1, 1])

# %%
strange = olr[0] > olr[3]
inversion = atm.temp_layer[0, :, 36:].max(axis=1) > atm.surface_temperature[0]
select = abs(atm.lat) <= 90

olr_avg = np.average(olr[:, select], axis=1, weights=weight[select])
print('OLR:')
print(f'effect total continuum: {np.round(olr_avg[0]-olr_avg[3], 1)} W m-2')
print(f'effect self continuum: {np.round(olr_avg[1]-olr_avg[3], 1)} W m-2')
print(f'effect foreign continuum: {np.round(olr_avg[2]-olr_avg[3], 1)} W m-2')

sdr_avg = np.average(sdr[:, select], axis=1, weights=weight[select])
print('SDR:')
print(f'effect total continuum: {np.round(sdr_avg[0]-sdr_avg[3], 1)} W m-2')
print(f'effect self continuum: {np.round(sdr_avg[1]-sdr_avg[3], 1)} W m-2')
print(f'effect foreign continuum: {np.round(sdr_avg[2]-sdr_avg[3], 1)} W m-2')


# %%
select = abs(atm.lat) <= 90
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax = spectral_plot(ax)
ax.set_title('change in OLR caused by WV continuum', fontsize=20)

colors = plt.get_cmap('coolwarm')
quantity = atm.surface_temperature[0]
# quantity = iwv
colorlist = colors((quantity - quantity.min())/
                   (quantity.max() - quantity.min()))

for i in range(len(colorlist)):
    if select[i]:
        ax.plot(wavenumber,
                mov_avg(toa_up_spectral[0][i].T - toa_up_spectral[3][i].T, 20)*1e3,
                color=colorlist[i])
ax.plot(wavenumber,
        np.average(
            mov_avg(toa_up_spectral[0][select].T - toa_up_spectral[3][select].T, 20)*1e3, 
            axis=1, weights=weight[select]), color='k', linewidth=3)
# cbar = plt.colorbar(colors)
# ax.set_ylim([-15,5])

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax = spectral_plot(ax)
ax.set_title('global mean change in OLR caused by WV continuum',
             fontsize=20)
ax.plot(wavenumber, 
        mov_avg(
            np.average(toa_up_spectral[0] - toa_up_spectral[3], axis=0, weights=weight),
            20)*1e3, 
        label='total')
ax.plot(wavenumber, 
        mov_avg(
            np.average(toa_up_spectral[1] - toa_up_spectral[3], axis=0, weights=weight),
            20)*1e3, 
        label='self')
ax.plot(wavenumber, 
        mov_avg(
            np.average(toa_up_spectral[2] - toa_up_spectral[3], axis=0, weights=weight),
            20)*1e3, 
        label='foreign')
ax.legend(fontsize=18)

# %%
# contingency table
# inversions_strange = len(np.where(strange[inversion])[0])
# inversions_normal = len(np.where(~strange[inversion])[0])
# not_inversions_strange = len(np.where(strange[~inversion])[0])
# not_inversions_normal = len(np.where(~strange[~inversion])[0])

# n = inversions_normal + inversions_strange \
#     + not_inversions_normal + not_inversions_strange
    
# hit_rate = inversions_strange / (inversions_strange + not_inversions_strange)
# false_alarm_rate = inversions_normal / (inversions_normal + not_inversions_normal)
# peirce_skill_score = hit_rate - false_alarm_rate

# print(np.round(peirce_skill_score, 2))

# %% reproduce Paynter & Ramaswamy (2012)

dict = {'OLR': olr,
        'SDR': sdr}

for var in dict.keys():
    # select = ~inversion
    select = atm.surface_temperature[0] > 0
    print(f"{len(atm.lat[select])}/100 sites selected")

    fig, ax = plt.subplots(2, 2, figsize=(20, 13))
    colormap = plt.get_cmap('inferno')
    color = atm.surface_temperature[0]

    ax[0, 0].set_ylim([dict[var][3][select].min(), dict[var][3][select].max()])

    ax[0, 0].set_title(f'{var} with no Continuum', fontsize=20)
    a = ax[0, 0].scatter(iwv[select], dict[var][3][select], 
                    c=np.array(color)[select], s=200,
                    vmin=284, vmax=302, cmap=colormap)
    ax[0, 1].set_title(f'Reduction in {var} due to Continuum', fontsize=20)
    a = ax[0, 1].scatter(iwv[select], -(dict[var][0] - dict[var][3])[select], 
                    c=np.array(color)[select], s=200,
                    vmin=284, vmax=302, cmap=colormap)
    ax[1, 0].set_title(f'Reduction in {var} due to Self Continuum', fontsize=20)
    a = ax[1, 0].scatter(iwv[select], -(dict[var][1] - dict[var][3])[select], 
                    c=np.array(color)[select], s=200,
                    vmin=284, vmax=302, cmap=colormap)
    ax[1, 1].set_title(f'Reduction in {var} due to Foreign Continuum', fontsize=20)
    a = ax[1, 1].scatter(iwv[select], -(dict[var][2] - dict[var][3])[select], 
                    c=np.array(color)[select], s=200,
                    vmin=284, vmax=302, cmap=colormap)

    ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1] = iwv_plot(ax[0, 0]),\
        iwv_plot(ax[0, 1]), iwv_plot(ax[1, 0]), iwv_plot(ax[1, 1])


    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    # [left, bottom, width, height]
    cax = plt.axes([0.92, 0.2, 0.03, 0.6])
    cb = plt.colorbar(a, cax=cax)
    cb.set_ticks(np.arange(284, 303, 2, dtype='int'))
    cb.set_ticklabels(cb.get_ticks(), fontsize=16)
    cb.set_label('surface temperature / K', fontsize=18)

    fig.savefig(f'/Users/froemer/Documents/wv_continuum/rfmip/plots/continuum_change_{var}.png',
                dpi=300)

# %%
select = abs(atm.lat) <= 90

fig, ax = plt.subplots(2, 2, figsize=(18, 12))

ax[0, 0].set_title('change in OLR caused by self continuum', fontsize=20)
ax[0, 1].set_title('change in OLR caused by foreign continuum', fontsize=20)
ax[1, 0].set_title('change in SDR caused by self continuum', fontsize=20)
ax[1, 1].set_title('change in SDR caused by foreign continuum', fontsize=20)

# ax[0, 0].set_yticks(np.arange(0, 50, 5, dtype='int'))
# ax[0, 0].set_yticklabels(ax[0, 0].get_yticks(), fontsize=14)
# ax[0, 1].set_yticks(np.arange(0, 14, 2, dtype='int'))
# ax[0, 1].set_yticklabels(ax[0, 1].get_yticks(), fontsize=14)

# cmap = 'jet'
cmap = 'density'
colors = plt.cm.get_cmap(cmap)
quantity = iwv

for i in range(len(colorlist)):
    if select[i]:
        ax[0, 0].scatter(wavenumber,
                -mov_avg(toa_up_spectral[1][i].T - toa_up_spectral[3][i].T, 20)*1e3,
                c=np.tile(quantity[i], len(wavenumber)),
                vmin=-10, vmax=60, s=1, cmap=colors)
        ax[0, 1].scatter(wavenumber,
                -mov_avg(toa_up_spectral[2][i].T - toa_up_spectral[3][i].T, 20)*1e3,
                c=np.tile(quantity[i], len(wavenumber)),
                vmin=-10, vmax=60, s=1, cmap=colors)
        ax[1, 0].scatter(wavenumber,
                -mov_avg(sfc_down_spectral[1][i].T - sfc_down_spectral[3][i].T, 20)*1e3,
                c=np.tile(quantity[i], len(wavenumber)),
                vmin=-10, vmax=60, s=1, cmap=colors)
        a = ax[1, 1].scatter(wavenumber,
                -mov_avg(sfc_down_spectral[2][i].T - sfc_down_spectral[3][i].T, 20)*1e3,
                c=np.tile(quantity[i], len(wavenumber)),
                vmin=-10, vmax=60, s=1, cmap=colors)

ax[0, 0], ax[0, 1] = spectral_plot(ax[0, 0]), spectral_plot(ax[0, 1])
ax[1, 0], ax[1, 1] = spectral_plot(ax[1, 0]), spectral_plot(ax[1, 1])


plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
cax = plt.axes([0.92, 0.1, 0.02, 0.8]) # [left, bottom, width, height]
cb = plt.colorbar(a, cax=cax)
cb.set_ticks(np.arange(0, 51, 10, dtype='int'),)
cb.set_ticklabels(cb.get_ticks(), fontsize=16)
cb.set_label('WVC / kg m-2', fontsize=18)

fig.savefig(f'/Users/froemer/Documents/wv_continuum/rfmip/plots/continuum_spectral_change_{cmap}.png',
            dpi=300)
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(np.average(irradiance[0, :, :, 1], axis=0, weights=weight),
        np.average(atm.pres_layer, axis=0, weights=weight)[::-1],
        label='upwelling')
ax.plot(-np.average(irradiance[0, :, :, 0], axis=0, weights=weight),
        np.average(atm.pres_layer, axis=0, weights=weight)[::-1],
        label='downwelling')

ax.invert_yaxis()
ax.legend(fontsize=18)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(np.average(heating_rates[0], axis=0, weights=weight),
        np.average(atm.pres_level, axis=0, weights=weight)[1:-1],
        label='heating rate')

ax.invert_yaxis()
ax.legend(fontsize=18)
# ax.set_xlim([-1, 1])
# ax.set_yscale('log')
# %%
fig, ax = plt.subplots(2, 2, figsize=(15, 15))

ax[0, 0].set_ylim([1000, 10])
ax[0, 1].set_ylim([1000, 10])
ax[1, 0].set_ylim([1000, 10])
ax[1, 1].set_ylim([1000, 10])

for i in range(atm.pres_level[:, 2:-2].shape[1]):
    a = ax[0, 0].scatter(atm.lat, atm.pres_level[:, i]/100, 
                     c=heating_rates[0, :, i],
                     s=50, cmap='RdYlBu', vmin=-3, vmax=0)  
    b = ax[0, 1].scatter(atm.lat, atm.pres_level[:, i]/100, 
                     c=(heating_rates[0, :, i]-heating_rates[3, :, i]),
                     s=50, cmap='RdYlBu', vmin=-1.2, vmax=0)
    c = ax[1, 0].scatter(atm.lat, atm.pres_level[:, i]/100, 
                     c=(heating_rates[1, :, i]-heating_rates[3, :, i]),
                     s=50, cmap='RdYlBu', vmin=-1.2, vmax=0)
    d = ax[1, 1].scatter(atm.lat, atm.pres_level[:, i]/100, 
                     c=(heating_rates[2, :, i]-heating_rates[3, :, i]),
                     s=50, cmap='RdYlBu', vmin=-0.1, vmax=0.1)


    
# %%
