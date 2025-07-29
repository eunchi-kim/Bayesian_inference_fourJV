from cadet import H5
import re
import numpy as np
import pandas as pd

'''
functions related to import
'''

def load_training_data(train_path):
    '''
    load_training_data : loads training data from hdf5

    Parameters 
    ----------
    train_path : file path to training data file

    returns
    -------
    ub : upper boundary for parameter variation
    lb : lower boundary of parameter variation
    n_var : number of variable parameters
    y1_max : maximum absolute value of the first data set (here: shifted jv curves)
    y2_min : minimum absolute value of the first data set (here: shifted jv curves)
    y2_max : maximum of second training data set (here: jsc values)
    y2_min : minimum of second training data set (here: jsc values)

    '''
    td = H5()   # create h5 object for the training data
    td.filename = train_path    # define filepath of the training data file
    td.load()   # load the training data hdf5 file
    ub = td.root.ub # extract the upper boundary of the variable parameters
    lb = td.root.lb # extract the lower boundary of the variable parameters
    n_var = td.root.n_var   # extract the number of variable parameters
    # extract the minimum and the maximum of the first training data set
    y1_max = td.root.y1_max 
    y1_min = td.root.y1_min
    # extract the minium and maximum of the second training data set
    y2_max = td.root.y2_max
    y2_min = td.root.y2_min
    
    return ub, lb, n_var, y1_max, y1_min, y2_max, y2_min

def load_exp_data(exp_path):
    '''
    load_exp_data : loads interpolated experimental data from hdf5

    Parameters 
    ----------
    exp_path : file path to experimental file

    returns
    -------
    y_exp_1 : y-axis of 1 or multiple illumination intensities of data set 1 (here: shifted jv curves)
    y_exp_2 : y-axis of 1 or multiple illumination intensities of data set 2 (here: jsc values)
    x_exp : x-axis of the JV curves

    '''

    exp_data = H5() # define h5 object for the experimental data
    exp_data.filename = exp_path.as_posix() # define filepath where experimental data is stored
    exp_data.load() # load experimental data hdf5 file
    y_exp_1 = exp_data.root.y1_exp  # extract data of first type (here: shifted jv curves)
    y_exp_2 = exp_data.root.y2_exp  # extract data of second type (here: jsc values)#
    x_exp = exp_data.root.x_exp  # extract x axis of first type (here: voltage values)
    return y_exp_1, y_exp_2, x_exp

def saveParas2csv(paras, name, csv_path):
    '''
    saveparas2csv : transform the paras data back into real units and save to csv file for Origin ploting
    
    Parameters
    ----------
    paras : characteristic parameters as a function of active layer thickness
    name : specifier to be printed as filename and comment line in origin
    sub_path : path to folder for saving


    Returns
    -------
    Nothing
    '''
    
    header_paras = [['Active layer thickness \i(d)','Short-circuit current density \i(J)\-(sc)', 'Open-circuit voltage \i(V)\-(oc)',
        'Fillfactor \i(FF)', 'Output-power density \i(P)\-(out)', 'Efficiency \i(\g(h))'],        
        ['nm', 'mA cm\+(-2)', 'V', '%', 'mW cm\+(-2)','%'], [name for nn in range(6)]]
    
    full_paras = pd.DataFrame()
    full_paras = pd.concat([full_paras,pd.DataFrame(header_paras)])
    full_paras = pd.concat([full_paras,pd.DataFrame(paras)])
    full_paras.to_csv(csv_path, header = False, index = False)
    del full_paras
    
def saveJV2csv(JV, name, csv_path):
    '''
    saveJV2csv : transform the JV data back into real units and save to csv file for Origin ploting
    
    Parameters
    ----------
    JV : voltage, current density [V, A/m²]
    name : specifier to be printed as filename and comment line in origin
    sub_path : path to folder for saving


    Returns
    -------
    Nothing
    '''
    
    header_JV = [['Voltage /i(V)', 'Shifted Current Density /i(J)-/i(J)/-(sc)', 'Current density /i(J)'],
                                 ['V','mA cm/+(-2)','mA cm/+(-2)'], [name for nn in range(3)]]
    
    full_JV = pd.DataFrame()
    full_JV = pd.concat([full_JV,pd.DataFrame(header_JV)])
    data_JV = pd.DataFrame(np.zeros((np.shape(JV)[0],3)))
    data_JV[0] = JV[:,0]
    data_JV[1] = (JV[:,1]-JV[0,1])/10           # mA/cm²
    data_JV[2] = JV[:,1]/10                     # mA/cm²
    full_JV = pd.concat([full_JV,data_JV])
    full_JV.to_csv(csv_path, header = False, index = False)
    del full_JV
    del data_JV