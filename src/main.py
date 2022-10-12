import os 
import sys
import pyarts

import write_xml_input_data as input_data
import experiment_setup as setup
import batch_calc as calc
import data_processing as post_pro
import data_visualisation as vis
from batch_lookuptable import BatchLookUpTable

def main(continua=True):
    # Read the experiment setup manuell in this script
    if len(sys.argv) == 1:
        exp_setups = [setup.read_exp_setup(exp_name='olr', 
                                           path='/Users/froemer/Documents/wv_continuum/rfmip/experiment_setups/')]
    else:
        exp_setups = []
        for arg in sys.argv[1:]:
            exp_setup_path = f'{os.getcwd()}/experiment_setups/'
            exp_setups.append(setup.read_exp_setup(exp_name=str(arg), path=exp_setup_path))

    for exp in exp_setups:
        run_pipline(exp, continua=continua)


def run_pipline(exp_setup, continua=True):
    simulation(exp_setup=exp_setup, continua=continua)
    postprocessing(exp_setup=exp_setup, continua=continua)


def simulation(exp_setup, continua=True) -> None: 
    # # Create input data
    # print('Create input data')
    # input_data.create_input_data(exp_setup)

    print('LookUpTable')
    lut = BatchLookUpTable(exp_setup=exp_setup)
    lut.calculate(optimise_speed=True, continua=continua)

    # Calculation
    print('Calculation')
    calc.run_arts_batch(exp_setup, continua=continua)


def postprocessing(exp_setup, continua=True):
    # Postprocessing
    # print('Postprocessing')
    data = post_pro.read_spectral_irradiance(exp_setup, continua=continua)
    heights = post_pro.read_heights(exp_setup)
    combined_data = post_pro.combine_sites(data, exp_setup)

    select = [20, 29, 39]  # select profiles for polar, mid-latitudes, and tropics
    selected_data, selected_heigths = data[select], heights[select]

    post_pro.save_data(combined_data, exp_setup, "combined_spectral_irradiance", continua=continua)
    post_pro.save_data(selected_data, exp_setup, "selected_spectral_irradiance", continua=continua)
    post_pro.save_data(selected_heigths, exp_setup, "selected_heights", continua=continua)

    # # Visualisation
    print('Visualisation')
    vis.plot_olr(exp_setup=exp_setup, continua=continua)


if __name__ == '__main__':
    main(continua=True)
#%%