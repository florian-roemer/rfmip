# %%
import pyarts
from batch_lookuptable import BatchLookUpTable

lut_on = pyarts.xml.load('../lookup_tables/continua_True/olr.xml')
lut_self = pyarts.xml.load('../lookup_tables/continua_self/olr.xml')
lut_foreign = pyarts.xml.load('../lookup_tables/continua_foreign/olr.xml')
lut_off = pyarts.xml.load('../lookup_tables/continua_False/olr.xml')

p_grid = lut_on.p_grid

# xsec: the absorption cross-sections (Tensor4; dimension: 
# [number of temperature perturbations, 
# number of species (and non-linear species perturbations), 
# number of frequencies, 
# number of pressure levels])

xsec_on = lut_on.xsec
xsec_self = lut_self.xsec
xsec_foreign = lut_foreign.xsec
xsec_off = lut_off.xsec

xsec_total_continuum = xsec_on - xsec_off
xsec_self_continuum = xsec_on - xsec_foreign
xsec_foreign_continuum = xsec_on - xsec_self

xsec_total_90 = xsec_off + 0.9*xsec_total_continuum
xsec_total_110 = xsec_off + 1.1*xsec_total_continuum

# %%
lut_total_90 = lut_on
lut_total_110 = lut_on

lut_total_90.xsec += 1
lut_total_110.xsec = xsec_total_110

# %%

