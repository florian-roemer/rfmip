
"""Simulate and plot Earth's outgoing longwave radiation (OLR)."""
# %%
import pyarts.workspace
from typhon import physics as phys
import matplotlib.pyplot as plt
from data_visualisation import convert_units
from experiment_setup import read_exp_setup
import numpy as np
import os

def calc_olr(atmfield,
             nstreams=10,
             fnum=300,
             fmin=1.0,
             fmax=75e12,
             species='default',
             verbosity=0,
             scale_species='H2O-SelfContCKDMT350',
             scale_factor=0.0):
    """Calculate the outgoing-longwave radiation for a given atmosphere.
    Parameters:
        atmfield (GriddedField4): Atmosphere field.
        nstreams (int): Even number of streams to integrate the radiative fluxes.
        fnum (int): Number of points in frequency grid.
        fmin (float): Lower frequency limit [Hz].
        fmax (float): Upper frequency limit [Hz].
        species (List of strings): List fo absorption species. Defaults to "default"
        verbosity (int): Reporting levels between 0 (only error messages)
            and 3 (everything).
    Returns:
        ndarray, ndarray: Frequency grid [Hz], OLR [Wm^-2]
    """
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.water_p_eq_agendaSet()
    ws.gas_scattering_agendaSet()
    ws.PlanetSet(option="Earth")

    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # Definition of species
    if species == 'default':
        ws.abs_speciesSet(species=[
            "H2O", "H2O-SelfContCKDMT350", "H2O-ForeignContCKDMT350",
            "CO2, CO2-CKDMT252",
        ])
    else:
        ws.abs_speciesSet(species=species)

    # Read line catalog
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="/Users/froemer/Documents/arts-cat-data/lines/")

    # Read cross section data
    # ws.ReadXsecData(basename="xsec/")

    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    @pyarts.workspace.arts_agenda(ws=ws, set_agenda=True)
    def propmat_clearsky_agenda(ws):
        ws.propmat_clearskyInit()
        ws.propmat_clearskyAddPredefined() # why does AddConts not work??
        ws.propmat_clearskyAddLines()
        ws.propmat_clearskyAddScaledSpecies(target=scale_species, 
                                            scale=scale_factor)


    # Calculate absorption
    # ws.propmat_clearsky_agendaAuto()
    ws.propmat_clearsky_agenda = propmat_clearsky_agenda

    # Weakly reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)

    # Atmosphere and surface
    ws.atm_fields_compact = atmfield
    ws.AtmosphereSet1D()
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    # Set surface height and temperature equal to the lowest atmosphere level
    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.surface_skin_t = ws.t_field.value[0, 0, 0]

    # Output radiance not converted
    ws.StringSet(ws.iy_unit, "1")

    # set cloudbox to full atmosphere
    ws.cloudboxSetFullAtm()

    # set particle scattering to zero, because we want only clear sky
    ws.scat_data_checked = 1
    ws.Touch(ws.scat_data)
    ws.pnd_fieldZero()

    # No sensor properties
    ws.sensorOff()

    # No jacobian calculations
    ws.jacobianOff()

    # Check model atmosphere
    ws.scat_data_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.lbl_checkedCalc()

    ws.propmat_clearsky_agenda_checkedCalc()

    # Perform RT calculations
    ws.DisortCalcIrradiance(nstreams=nstreams, emission=1)

    olr = ws.spectral_irradiance_field.value[:, -1, 0, 0, 1][:]

    return ws.f_grid.value[:], olr


os.chdir('/Users/froemer/Documents/wv_continuum/rfmip')

exp_setup = read_exp_setup(exp_name='olr', 
                           path='/Users/froemer/Documents/wv_continuum/rfmip/experiment_setups/')

fnum = 300

atmfield = pyarts.xml.load('input/olr/atm_fields.xml')[0]
fgrid, olr_self100 = calc_olr(atmfield, fnum=fnum, scale_species="H2O-SelfContCKDMT350", scale_factor=0.0)
fgrid, olr_self90 = calc_olr(atmfield, fnum=fnum, scale_species="H2O-SelfContCKDMT350", scale_factor=-0.1)

wavenumber = np.linspace(exp_setup.spectral_grid['min'],
                         exp_setup.spectral_grid['max'],
                         fnum,
                         endpoint=True)

_, olr_self100 = convert_units(exp_setup, wavenumber, olr_self100)
_, olr_self90 = convert_units(exp_setup, wavenumber, olr_self90)


fgrid, olr_foreign100 = calc_olr(atmfield, fnum=fnum, scale_species="H2O-ForeignContCKDMT350", scale_factor=0.0)
fgrid, olr_foreign90 = calc_olr(atmfield, fnum=fnum, scale_species="H2O-ForeignContCKDMT350", scale_factor=-0.1)

_, olr_foreign100 = convert_units(exp_setup, wavenumber, olr_foreign100)
_, olr_foreign90 = convert_units(exp_setup, wavenumber, olr_foreign90)

# %%
plt.plot(wavenumber, 1e3*(olr_self90-olr_self100), label='self -10%')
plt.plot(wavenumber, 1e3*(olr_foreign90-olr_foreign100), label='foreign -10%')
plt.legend()

# %%
