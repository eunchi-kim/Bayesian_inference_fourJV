import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

'''
functions for converting the scale of parameter space (to log or back to linscale...)
'''

def log_trans(paras, n_var, ln_idx):
    '''
    log_trans : take the logarithm of a certain subset of data that is varied over several orders of magnitude  
                !!! number of parameters that scales linearly must be defined here !!!
    
    Parameters
    ----------
    paras : numpy.array of data of which a subset is supposed to be transformed. one dimension must be of the size n_var 
    n_var : number of variable parameters
    ln_idx : index of the start of logscale parameter

    Returns
    -------
    mod : data with the natural logarithm of some data
    ''' 
    if np.shape(paras)[0] == n_var:  # check which dimension contains the n_var parameters
        mod = np.concatenate((paras[:ln_idx],np.log(paras[ln_idx:])),axis=0)    # take logarithm of some parameters along that axis
    elif np.shape(paras)[1] == n_var:    # check which dimension contains the n_var parameters
        mod = np.concatenate((paras[:,:ln_idx],np.log(paras[:,ln_idx:])),axis=1)     # take logarithm of some parameters along that axis
    return mod

def exp_trans(paras, n_var, ln_idx):
    '''
    exp_trans : take the exponential of a certain subset of data that is varied over several orders of magnitude  
                !!! number of parameters that scales linearly must be defined here !!!
    
    Parameters
    ----------
    paras : numpy.array of data of which a subset is supposed to be transformed. one dimension must be of the size n_var 
    n_var : number of variable parameters
    ln_idx : index of the start of logscale parameter

    Returns
    -------
    mod : data with the exponential of some data
    ''' 
    if np.shape(paras)[0] == n_var:  # check which dimension contains the n_var parameters
        mod = np.concatenate((paras[:ln_idx],np.exp(paras[ln_idx:])),axis=0)    # take exponential of some parameters along that axis
    elif np.shape(paras)[1] == n_var:  # check which dimension contains the n_var parameters
        mod = np.concatenate((paras[:,:ln_idx],np.exp(paras[:,ln_idx:])),axis=1)    # take exponential of some parameters along that axis
    return mod

def log10_trans(paras, n_var, ln_idx):
    '''
    log10_trans : take the logarithm to the base 10 of a certain subset of data that is varied over several orders of magnitude  
                !!! number of parameters that scales linearly must be defined here !!!
    
    Parameters
    ----------
    paras : numpy.array of data of which a subset is supposed to be transformed. one dimension must be of the size n_var 
    n_var : number of variable parameters
    ln_idx : index of the start of logscale parameter

    Returns
    -------
    mod : data with the log10 of some data
    ''' 
    if np.shape(paras)[0] == n_var:
        mod = np.concatenate((paras[:ln_idx],np.log10(paras[ln_idx:])),axis=0)
    elif np.shape(paras)[1] == n_var:
        mod = np.concatenate((paras[:,:ln_idx],np.log10(paras[:,ln_idx:])),axis=1)
    return mod

def power10_trans(paras, n_var, ln_idx):
    '''
    power10_trans : make the 10 to the power of a certain subset of data that is varied over several orders of magnitude  
                !!! number of parameters that scales linearly must be defined here !!!
    
    Parameters
    ----------
    paras : numpy.array of data of which a subset is supposed to be transformed. one dimension must be of the size n_var 
    n_var : number of variable parameters
    ln_idx : index of the start of logscale parameter

    Returns
    -------
    mod : data with the log10 of some data
    ''' 
    if np.shape(paras)[0] == n_var:
        mod = np.concatenate((paras[:ln_idx],np.power(10,paras[ln_idx:])),axis=0)
    elif np.shape(paras)[1] == n_var:
        mod = np.concatenate((paras[:,:ln_idx],np.power(10,paras[:,ln_idx:])),axis=1)
    return mod

'''
functions related to JV data
'''

def transform_JV_to_original(y_exp_1,y_exp_2, NNpoints,oneJVlen):
    '''
    transform_JV_to_original: combine log(J+Jsc) and Jsc data together and transform it to the original JV data
    
    Parameters
    ----------
    y_exp_1 : an array of log(J+Jsc)
    y_exp_2 : an array of log(Jsc)
    
    returns
    -------
    y_exp : array of J 
    
    '''
    y_exp_1_actual = np.exp(y_exp_1)          
    y_exp_2_actual = np.exp(y_exp_2)           
    y_exp = np.zeros(NNpoints)
    for ii in range(NNpoints//oneJVlen):
        if np.shape(y_exp_1)[0] == NNpoints:
            y_exp[ii*oneJVlen:(ii+1)*oneJVlen] = y_exp_1_actual[ii*oneJVlen:(ii+1)*oneJVlen] - y_exp_2_actual[ii]
        elif np.shape(y_exp_1)[1] == NNpoints:
            y_exp[ii*oneJVlen:(ii+1)*oneJVlen] = y_exp_1_actual[0,ii*oneJVlen:(ii+1)*oneJVlen] - y_exp_2_actual[0,ii]
    return y_exp

def scale_and_exponentiate(pred, min_val, max_val):
    return np.exp(pred * (max_val - min_val) + min_val)

def scale_back(pred, min_val, max_val):
    return pred * (max_val - min_val) + min_val

def get_sigma_exp(y_exp,x_exp,oneJVlen,NNpoints,phis,std_phi,std_vol):
    # Calculate uncertainty of measurement
    x_exp_1 = x_exp[:oneJVlen,]
    y_exp_sigma = y_exp.reshape((NNpoints//oneJVlen,oneJVlen))

    dJ_dphi = np.gradient(y_exp_sigma,phis,axis=0,edge_order=2)     # shape (4,128)
    sigma_phi = phis * std_phi
    phi_term = np.zeros(dJ_dphi.shape)
    for phi_idx in range(dJ_dphi.shape[0]):
        phi_term[phi_idx,:]=dJ_dphi[phi_idx,:]*sigma_phi[phi_idx]
    dJ_dV = np.gradient(y_exp_sigma,x_exp_1,axis=-1,edge_order=2)  # shape (4,128)
    sigma_vol = std_vol           

    sigma_J = np.sqrt((phi_term)**2+(dJ_dV*sigma_vol)**2)
    del y_exp_sigma, dJ_dphi, dJ_dV, phi_term, phis
    sigma_J = np.average(sigma_J,axis=-1)
    return sigma_J

'''
functions related to monte-carlo integration
'''
def getRange(ga_y_array,ga_x_array,threshold):
    nb_params = ga_x_array.shape[-1]
    ga_array_mask = ga_y_array>np.percentile(ga_y_array,threshold)
    reduced_ga_x_array = ga_x_array[ga_array_mask,:]
    new_lb_mod = np.zeros((nb_params))
    new_ub_mod = np.zeros((nb_params))
    for comb in range(nb_params):
        new_ub_mod[comb] = np.max(reduced_ga_x_array[:,comb])
        new_lb_mod[comb] = np.min(reduced_ga_x_array[:,comb])
    return new_lb_mod, new_ub_mod

def plot_mc_result(mc_inte_result: dict, **kwargs):
    '''
    Plot histograms of probabilistic integral values grouped by volume.

    Args:
        mc_inte_result (dict): A dictionary where integral results are stored.
                               Each key maps to a dictionary containing:
                               - 'mc_inte_all': Array of probabilistic integral values.
                               - 'vol': Volume percentage as a float.
        kwargs (dict): Additional options for plotting:
            - plot_figure (bool): Whether to show the plot. Default is True.
            - save_figure (bool): Whether to save the figure. Default is False.
            - filename (Path): Filepath to save the figure if `save_figure` is True.
            - figsize (list or array): size of the figure

    '''
    # Default options
    plot_figure = kwargs.get('plot_figure', True)
    save_figure = kwargs.get('save_figure', False)
    filename = kwargs.get('filename', 'mc_inte.png')
    figsize = kwargs.get('figsize', [5,12])
    
    # Find global min and max of integral values
    all_values = []
    for key in mc_inte_result.keys():
        all_values.extend(mc_inte_result[key]['mc_inte_all'])
    all_values = np.array(all_values)
    
    mc_inte_min = all_values.min()
    mc_inte_max = all_values.max()
    
    if list(mc_inte_result.keys())[0][0] == 't':
        label_key = 'vol'
        xlabel = 'volume'
        xunit = '%'
        convert2perc = 100
    elif list(mc_inte_result.keys())[0][0] == 's':
        label_key = 'seq_num'
        xlabel = 'number of points'
        xunit = ''
        convert2perc = 1
    
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])
    # plot histogram 
    for key in mc_inte_result.keys():
        ax[0].hist(mc_inte_result[key]['mc_inte_all'],label=f"{label_key}: {int(mc_inte_result[key][label_key]*convert2perc)} {xunit}",bins=np.linspace(mc_inte_min-0.001,mc_inte_max+0.001))
    ax[0].legend()
    ax[0].set_ylabel("frequency")
    ax[0].set_xlabel("Integral")
    
    # y-errorbar plot
    ax[1].errorbar(np.array([mc_inte_result[key][label_key] for key in mc_inte_result.keys()])*convert2perc,
             [mc_inte_result[key]['mean'] for key in mc_inte_result.keys()],[mc_inte_result[key]['std'] for key in mc_inte_result.keys()])

    ax[1].set_xlabel(xlabel + ' (' + xunit + ')')
    ax[1].set_ylabel("mean & std of integral")

     # Save the figure if required
    if save_figure:
        fig.savefig(filename)
    
    # Show the figure if required
    if plot_figure:
        plt.show()
    else:
        plt.close()
        
def save_mc_result(mc_inte_result: dict, x_type: str, **kwargs):
    '''
    x_type (str): 'vol' or 'seq_num'
    '''
    # Default options
    filename = kwargs.get('filename', 'mc_inte.csv')
    
    if x_type == 'vol':
        convert2per = 100
    elif x_type == 'seq_num':
        convert2per = 1
    
    x_array = (np.array([mc_inte_result[key][x_type] for key in mc_inte_result.keys()])*convert2per).reshape(-1,1)
    mean_integral = np.array([mc_inte_result[key]['mean'] for key in mc_inte_result.keys()]).reshape(-1,1)
    std_integral = np.array([mc_inte_result[key]['std'] for key in mc_inte_result.keys()]).reshape(-1,1)
    integral_all = np.vstack([mc_inte_result[key]['mc_inte_all'] for key in mc_inte_result.keys()])
    
    data = pd.DataFrame(np.concatenate((x_array,mean_integral,std_integral),axis=-1))   
    if x_type == 'vol':
        header = [['Volume', 'mean', 'variance'],
                  ['%','','']]
    elif x_type == 'seq_num':
        header = [['number of points', 'mean', 'variance'],
                  ['','','']]
    df = pd.DataFrame(pd.concat([pd.DataFrame(header),data]))
    df.to_csv(filename, index=False,header=False)
        
    index = str(filename).find('.csv')
    filename_raw = Path(str(filename)[:index] + '_raw.csv')     # if filename is xxx.csv, convert to xxx_raw.csv
    data_all = pd.DataFrame(np.concatenate((x_array,integral_all), axis=-1))
    if x_type == 'vol':
        header = [['Volume', 'integral'],
                  ['%','']]
    elif x_type == 'seq_num':
        header = [['number of points', 'integral'],
                  ['','']]
    mc_inte_logLH_allrd_df = pd.DataFrame(pd.concat([pd.DataFrame(header),data_all]))
    mc_inte_logLH_allrd_df.to_csv(filename_raw, index=False, header=False)
    
'''
Others
'''

def make_folder(foldername: Path):
    if not foldername.exists():
        foldername.mkdir()
    else:
        print('the folder already exists!')
        
def power(x, y):
    """
    Element-wise power operation.

    Parameters
    ----------
    x : array_like
        Base numbers.
    y : array_like
        Exponents.

    Returns
    -------
    out : ndarray
        The result of the power operation.

    See Also
    --------
    numpy.power
    """
    return np.exp(y * np.log(x))