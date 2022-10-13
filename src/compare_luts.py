# %%
import pyarts

lut_on = pyarts.xml.load('lookup_tables/continua_True/olr.xml')
lut_off = pyarts.xml.load('lookup_tables/continua_False/olr.xml')

p_grid = lut_on.p_grid

xsec_on = lut_on.xsec
xsec_off = lut_off.xsec
