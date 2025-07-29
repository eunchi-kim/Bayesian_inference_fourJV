import tensorflow as tf
import joblib
import numpy as np
from pathlib import Path
from typing import Union
from scipy.special import logsumexp
import matplotlib
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import jstyleson
import pickle
import matplotlib.cm as cm

# pacakges defined by myself
import post_pdf.import_funcs as Imf
import post_pdf.para_est_utils as utils
import post_pdf.cmaes as cmaes


class posterior_pdf():
    def __init__(self,
                 reg1_path: Path=None
                 ,reg2_path: Path=None
                 ,scaler_path: Path=None
                 ,train_path: Path=None
                 ,ln_idx: int=None ,NNpoints: int = 512,oneJVlen: int = 128):
        '''
        Args:
        
        reg1_path (Path): Path to the first neural network regressor model.
        reg2_path (Path): Path to the second neural network regressor model.
        scaler_path (Path): Path to the standard scaler file used for data normalization.
        train_path (Path): Path to the training data file for loading bounds and variable information.
        ln_idx (int): Indices specifying which parameters should undergo a logarithmic transformation. (starting index)
        NNpoints (int, optional): Number of data points for neural network computations.
        oneJVlen (int, optional): Length of a single JV curve, specific to the problem setup.
        '''
        
        self.reg1_path = reg1_path
        self.reg2_path = reg2_path
        self.scaler_path = scaler_path
        self.train_path = train_path
        self.ln_idx = ln_idx
        self.NNpoints = NNpoints
        self.oneJVlen = oneJVlen
        
        self.reg1, self.reg2, self.stscaler = None, None, None
        self.lb, self.ub, self.n_var, self.y1_max, self.y1_min, self.y2_max, self.y2_min = None, None, None, None, None, None, None
        
        # import NN regressors and relevant data
        if not reg1_path == None:
            self._load_models()
        
        # Load models and data
        if not train_path==None:
            self._load_training_data()

            ## very specific for my problem ## 
            self._adjust_bounds_for_problem()
            
            # Get log transformed boundaries
            self.lb_mod = utils.log_trans(self.lb, self.n_var, self.ln_idx)
            self.ub_mod = utils.log_trans(self.ub, self.n_var, self.ln_idx)
        
    def _load_models(self):
        ''' Load neural network regressors and scaler. '''
        try:
            self.reg1 = tf.keras.models.load_model(self.reg1_path)
            self.reg2 = tf.keras.models.load_model(self.reg2_path)
            self.stscaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise RuntimeError(f"Error loading models or scaler: {e}")

    def _load_training_data(self):
        ''' Load training data and bounds. '''
        try:
            self.ub, self.lb, self.n_var, self.y1_max, self.y1_min, self.y2_max, self.y2_min = Imf.load_training_data(self.train_path)
        except Exception as e:
            raise RuntimeError(f"Error loading training data: {e}")
        
    def _adjust_bounds_for_problem(self):
        ''' Apply problem-specific adjustments to bounds. '''
        self.lb = np.delete(self.lb, [2, 5])
        self.ub = np.delete(self.ub, [2, 5])
        self.n_var -= 2
        
    def attach_simdata(self, gentype: str, param_val: np.ndarray, param_keys: np.ndarray, Rpdark: np.ndarray = np.array([1e+6,1e+6,1e+6,1e+6])):
        '''
        Attach simulated data to the object for analysis.

        Parameters:
            gentype -> 'NN'
                select which by method to generate JV data
            param_val -> list or array-like
                List of parameter values used to generate the simulated data. In original form.
            param_keys -> list or array-like
                List of parameter names corresponding to `param_val`.
            Rpdark -> array-like, optional (default=1e+6)
                Resistance in the dark in the unit of Ohm m².
        '''  
        self.plot_true_values = True
        self.true_param_val = param_val
        self.true_param_keys = param_keys
        self.Rpdark = Rpdark
        if gentype == 'NN':
            curdir = Path().absolute()  # get current directory
            input_path = curdir / 'example_NNmodel' / 'inputs_for_training_data.jsonc'
            with open(input_path) as f: # open input file
                input = jstyleson.load(f) # read input file
            self.x_exp = np.tile(np.linspace(input["vstart"] + input["vstep"], input["vend"], self.oneJVlen),4)
            
            param_val = utils.log_trans(param_val,8,4) # specific to my problem  
            param_val = (self.stscaler).transform(param_val.reshape(1,-1))         # the input to the regressor should be 2d-array
            self.y_exp_1 = utils.scale_back((self.reg1).predict(param_val, verbose=0), self.y1_min, self.y1_max).ravel()
            self.y_exp_2 = utils.scale_back((self.reg2).predict(param_val, verbose=0), self.y2_min, self.y2_max).ravel()
            
        self.y_exp = utils.transform_JV_to_original(self.y_exp_1,self.y_exp_2,self.NNpoints,self.oneJVlen)
    
    def attach_expdata(self,exp_path: Path,Rpdark: np.ndarray = np.array([1e+6,1e+6,1e+6,1e+6])):
        '''
        Attach experimental data to the object for analysis.

        Parameters:
            exp_path -> pathlib.Path
                Path to the experimental data file.
            Rpdark -> array-like
                Dark resistance in the unit of Ohm m²
        '''
        self.plot_true_values = False
        self.Rpdark=Rpdark

        self.y_exp_1 ,self. y_exp_2, self.x_exp= Imf.load_exp_data(exp_path)
        self.y_exp = utils.transform_JV_to_original(self.y_exp_1,self.y_exp_2,self.NNpoints,self.oneJVlen)
        
    def save_pdf(self, filename: Path='post_pdf.pkl'):
        '''
        Save the current state of the object to a pickle file.
        '''
        # Tensorflow objects cannot be processed in pickle format, so I manually delete them when saving.
        pdf_temp = self
        del pdf_temp.reg1
        del pdf_temp.reg2
        with open(filename, 'wb') as f: 
            pickle.dump(pdf_temp.__dict__,f)
            
    def load_pdf(self, filename: Path):
        '''
        Load the state of the object from a pickle file.
        '''
        with open(filename, 'rb') as f:
            dataset = pickle.load(f) 
        for key, value in dataset.items():
            setattr(self, key, value)
        if not self.reg1_path == None:
            self._load_models()
        
        
    def calc_sigma_exp(self,phis: np.ndarray,std_phi: Union[int,float],std_volt: float):
        '''
        Calculate the experimental uncertainty (sigma) for parameter estimation.
        
        Parameters:
            phis -> array-like
                Array of light intensity values (mW/cm²)
            std_phi -> float
                Standard deviation of the phase values (unitless)
            std_volt -> float
                Standard deviation of the voltage values (V)
        '''
        self.phis = phis
        self.std_phi = std_phi
        self.std_volt = std_volt
            
        self.sigma_exp = utils.get_sigma_exp(self.y_exp,self.x_exp,self.oneJVlen,self.NNpoints,phis,std_phi,std_volt)

    def run_cmaes(self,obs_mask: np.ndarray[bool], shifterror: bool = True, width: int= 10, num_run: int=10, verbose: bool=False):
        '''
        very specific function for my problem
        change cmaes.run_cmaes_reduced_jv() and the corresponding eval_func to adjust to one's specific problem
        '''            
        self.obs_mask = obs_mask
        callbackdata=dict()
        error_fit=1e20
        self.width = width
        if shifterror:
            y1_exp = np.exp(self.y_exp_1).ravel()
            y2_exp = np.exp(self.y_exp_2).ravel()
            # y1_exp = ((self.y_exp_1 - self.y1_min)/(self.y1_max - self.y1_min)).ravel()
            # y2_exp = ((self.y_exp_2 - self.y2_min)/(self.y2_max - self.y2_min)).ravel()
            
        # run cmaes
        for zz in range(num_run):
            if shifterror: 
                best_scaled_temp, best_ln_temp, best_norm_temp, error_fit_temp, callbackdata[f"{zz}"] = cmaes.run_cmaes_reduced_jv_shifted(self.lb_mod, self.ub_mod, self.n_var,
                                            self.y_exp, y1_exp, y2_exp, self.x_exp, self.sigma_exp, 
                                            self.y1_max, self.y1_min, self.y2_max, self.y2_min, self.reg1, self.reg2, self.stscaler, obs_mask, self.Rpdark, self.width, verbose)
            else:
                best_scaled_temp, best_ln_temp, best_norm_temp, error_fit_temp, callbackdata[f"{zz}"] = cmaes.run_cmaes_reduced_jv(self.lb_mod, self.ub_mod, self.n_var,
                                            self.y_exp, self.x_exp, self.sigma_exp, 
                                            self.y1_max, self.y1_min, self.y2_max, self.y2_min, self.reg1, self.reg2, self.stscaler, obs_mask, self.Rpdark, verbose)    # run fitting routine
            # error_ga_all[zz] = error_fit_temp.item()
            if (error_fit_temp < error_fit):    # check if new error is better than old error
                # update the fitting results
                error_fit = error_fit_temp  
                best_scaled = best_scaled_temp
                best_ln = best_ln_temp                
                best_norm = best_norm_temp
        
        # store callbackdata    
        for zz in range(num_run):    
            for ii in range(len(callbackdata[f"{zz}"]["offspring"])):
                offspring_ln = (callbackdata[f"{zz}"]["offspring"][ii] * (self.ub_mod-self.lb_mod) + self.lb_mod)
                if ii==0 and zz==0:
                    ga_suggested_par_mat = offspring_ln
                else:
                    ga_suggested_par_mat = np.concatenate((ga_suggested_par_mat,offspring_ln),axis=0) 
        
        cmaes_result = {
            'error_fit': error_fit,
            'best_scaled': best_scaled,
            'best_ln': best_ln,
            'best_norm' : best_norm,
            'num_run': num_run
        }        
        
        if verbose:
            print('Starting log likelihood calculation')
        
        self.cmaes = cmaes_result
        log_LH_ga = self.calc_LH_NN(ga_suggested_par_mat)
        if verbose:
            print("Finished log likelihodd calculation")
        
        return ga_suggested_par_mat, log_LH_ga
        
    '''
    functions related to likelihood
    '''

    def calc_LH_NN(self,para_grid: np.ndarray,compute_size: Union[int,float] = 2e5):
        '''
        calculate log likelihood 
        para_grid (2d-array) : parameter combination
        compute_size (int or float, optional) : the size which the memory space can handle at once
        '''
        beta = 1/self.sigma_exp**2
        num_points = 1
        
        # check the shape of the para_grid
        if para_grid.ndim != 2:
            raise ValueError("Expected a 2D np array.")
        if para_grid.shape[0] == self.n_var:
            para_grid = para_grid.T
        
        if para_grid.shape[0] > compute_size:
            Xcc = np.array_split(para_grid, int(para_grid.shape[0]/compute_size))
            for ii in range(int(para_grid.shape[0]/compute_size)):
                if ii == 0:
                    error_predprime = self.errorJV_fromNN(Xcc[ii])
                    aa0prime = -beta/2 * num_points*error_predprime**2 - num_points/2*np.log(beta) - num_points/2*np.log(2*np.pi)
                else:
                    error_predprime = self.errorJV_fromNN(Xcc[ii])
                    aa0prime_temp = -beta/2 * num_points*error_predprime**2 - num_points/2*np.log(beta) - num_points/2*np.log(2*np.pi)
                    aa0prime = np.concatenate((aa0prime, aa0prime_temp),axis=0)
            aa1 = aa0prime
                
        else:
            error_pred = self.errorJV_fromNN(para_grid)
            aa1 = -beta/2 * num_points*error_pred**2 - num_points/2*np.log(beta) - num_points/2*np.log(2*np.pi)

        if np.count_nonzero(self.obs_mask)==1:
            post_aa = aa1[:,self.obs_mask]
        else:
            post_aa = np.sum(aa1[:,self.obs_mask],axis=-1)
        
        return post_aa.ravel()

    def errorJV_fromNN(self,theta: np.ndarray,NNpoints: int=512,oneJVlen: int=128):
        '''
        calculate the mean-squared-root error of all 4 JVs  
        theta (2d-array) : a combination of parameter values used for prediction
        NNpoints (int, optional) : the total number of data points generated by the neural network.
        oneJVlen (int, optional) : the number of data points in a single JV curve
        
        return
        errors (2d-array of shape (num_combinations,4)) : the MSRE values for each JV curve.
                Each row corresponds to a set of parameters, and each column represents a JV curve.
        '''
        below_Voc_mask = self.y_exp<0
        
        #### specific to my problem ####
        theta = np.insert(theta,2,1.3,axis=-1)
        theta = np.insert(theta,5,theta[:,4],axis=-1)
        #####################################
        
        theta_norm = (self.stscaler).transform(theta)
        y1 = (self.reg1).predict(theta_norm ,verbose=0)
        y2 = (self.reg2).predict(theta_norm ,verbose=0)
        
        y1_scaled = np.array(y1*(self.y1_max-self.y1_min)+self.y1_min)
        y2_scaled = np.array(y2*(self.y2_max-self.y2_min)+self.y2_min)
        y1_scaled = np.exp(y1_scaled)     # mA/cm²
        y2_scaled = np.exp(y2_scaled)     # mA/cm²
        
        # transformed in a original scale 
        for ii in range(NNpoints//oneJVlen):
            y1_scaled[:,ii*oneJVlen:(ii+1)*oneJVlen] = y1_scaled[:,ii*oneJVlen:(ii+1)*oneJVlen] - y2_scaled[:,ii].reshape((y2_scaled.shape[0],1))
        y1_scaled += self.x_exp/self.Rpdark[1] + self.x_exp**2/self.Rpdark[2]
        
        errors = list()
        for ii in range(NNpoints//oneJVlen):
            phi_mask = np.repeat([False],NNpoints)
            phi_mask[ii*oneJVlen:(ii+1)*oneJVlen] = below_Voc_mask[ii*oneJVlen:(ii+1)*oneJVlen]
            error = (y1_scaled[:,phi_mask]- self.y_exp[phi_mask])
            error = np.sqrt(np.mean(error**2, axis = -1))
            errors.append(error)

        errors = np.vstack(errors)
        return errors.T
    
    '''
    functions related to monte carlo integration
    '''
    def calc_mc_inte(self,
                    ga_suggested_par_mat: np.ndarray,
                    log_LH_ga: np.ndarray,
                    calc_type: str,
                    threshold_array: Union[list, np.ndarray, None] = None,
                    seq_num_array: Union[list, np.ndarray, None] = None,
                    seq_num_std: Union[int, float] = None,
                    threshold: Union[int, float] = None,
                    num_run: Union[int, float] = 10,
                    verbose: bool = False):
        '''
        Performs Monte Carlo (MC) integration over regions of the parameter space defined by either thresholds in the likelihood function
        or a fixed number of samples.
        MC integration can be either applied to approximation of marginal likelihood or relative entropy.

        Parameters:
            ga_suggested_par_mat (np.array): 2D-array containing parameter combinations generated by a genetic algorithm (GA).
            log_LH_ga (np.array): 1D array of log likelihood values corresponding to ga_suggested_par_mat.
            calc_type (str): 'margLH' or 'entropy'
            threshold_array (list or np.array, optional): Thresholds for the percentile of log likelihood, used to define regions.
            seq_num_array (list or np.array, optional): Array of fixed sample counts for evaluating integration.
            seq_num_std (int or float, optional): Standard number of samples scaled by the volume of the region (used with thresholds).
            threshold (int or float, optional): Threshold for the percentile of log likelihood, used to define regions.
            num_run (int or float, optional): Number of Monte Carlo runs for statistical evaluation of the integral. Defaults to 10.

        Returns:
            mc_inte_result (dict): Results of the MC integration for each threshold or sample count.
        '''
        if threshold_array is not None:
            threshold_array = sorted(threshold_array, reverse=True)
        
        mc_inte_result = dict()

        if threshold_array is not None:  # Perform MC integration for threshold-based regions
            for threshold in threshold_array:
                new_lb_mod, new_ub_mod = utils.getRange(log_LH_ga, ga_suggested_par_mat, threshold)
                vol = np.prod(new_ub_mod - new_lb_mod) / np.prod(self.ub_mod - self.lb_mod)
                seq_num = int(seq_num_std * vol)
                mc_inte_rd = np.zeros(num_run)

                for jj in range(num_run):
                    par_mat = np.random.uniform(0, 1, (self.n_var, seq_num)).T * (new_ub_mod - new_lb_mod) + new_lb_mod
                    log_LH = self.calc_LH_NN(par_mat)
                    if calc_type == 'margLH':
                        mc_inte_rd[jj] = np.exp(logsumexp(log_LH) - np.log(seq_num) + np.log(vol))
                    elif calc_type == 'entropy':
                        if self.margLH == []:
                            raise ValueError("approximation of marginal likelihood should be done in advance")
                        else:
                            log_prob = log_LH - np.log(self.margLH)
                            mc_inte_rd[jj] = np.mean(log_prob * np.exp(log_prob) / (1/vol))*vol

                var_temp = np.std(mc_inte_rd)
                mean_temp = np.mean(mc_inte_rd)
                mc_inte_result[f'threshold_{threshold}'] = {
                    'vol': vol,
                    'seq_num': seq_num,
                    'mc_inte_all': mc_inte_rd,
                    'mean': mean_temp,
                    'std': var_temp
                }

                if verbose:
                    print(f'Threshold {threshold} completed')

        elif seq_num_array is not None:  # Perform MC integration for fixed sample sizes
            new_lb_mod, new_ub_mod = utils.getRange(log_LH_ga, ga_suggested_par_mat, threshold) if threshold else (self.lb_mod, self.ub_mod)
            vol = np.prod(new_ub_mod - new_lb_mod) / np.prod(self.ub_mod - self.lb_mod)

            for seq_num in seq_num_array:
                mc_inte_rd = np.zeros(num_run)

                for jj in range(num_run):
                    par_mat = np.random.uniform(0, 1, (self.n_var, seq_num)).T * (new_ub_mod - new_lb_mod) + new_lb_mod
                    log_LH = self.calc_LH_NN(par_mat)
                    if calc_type == 'margLH':
                        mc_inte_rd[jj] = np.exp(logsumexp(log_LH) - np.log(seq_num) + np.log(vol))
                    elif calc_type == 'entropy':
                        if self.margLH == []:
                            raise ValueError("approximation of marginal likelihood should be done in advance")
                        else:
                            log_prob = log_LH - np.log(self.margLH)
                            mc_inte_rd[jj] = np.mean(log_prob * np.exp(log_prob) / (1/vol))*vol

                var_temp = np.std(mc_inte_rd)
                mean_temp = np.mean(mc_inte_rd)
                mc_inte_result[f'seq_num_{seq_num}'] = {
                    'vol': vol,
                    'seq_num': seq_num,
                    'mc_inte_all': mc_inte_rd,
                    'mean': mean_temp,
                    'std': var_temp
                }

        else:
            raise ValueError("Either threshold_array or seq_num_array must be provided.")
        
        # automatically set the evidence (self.margLH) to the result with the highest volume or largest sample size.
        if calc_type=='margLH':
            mc_keys = list(mc_inte_result.keys())
            self.margLH = mc_inte_result[mc_keys[-1]]['mean']

        return mc_inte_result
    
    def plot_conditionalpdf(self, 
                            label: Union[list, np.ndarray], 
                            header_csv: Union[list, np.ndarray]=None,**kwargs):
        '''
        Generates a grid of conditional probability density function (PDF) plots.
        The plot includes off-diagonal contour plots representing conditional PDFs for parameter pairs
        , and diagonal plots showing 1D marginal distributions for individual parameters.
        
        Parameters:
            label (list, arr): Labels for the parameters to be displayed on the axes of the plots.
            header_csv (list, arr, Optional): Labels and units for the parameters to be saved in .csv files.
            **kwargs (optional arguments):
                - vmin (float): Minimum log-probability value for contour plots, default is -25.
                - Nres (int): Resolution of the grid for parameter sampling, default is 50.
                - plot_figure (bool): If True, the figure is displayed interactively. Default is True.
                - save_figure (bool): If True, the figure is saved to a file. Default is False.
                - filename (path): Filename for saving the figure, default is 'conditionalpdf_bestfit.png'.
                - figsize (tuple): Dimensions of the figure, default is [12, 12].
                - save_csv (bool): If True, the computed LHS data is saved to a .csv file. Default is False.
                - csv_filename (str): Filename for the .csv file, default is 'conditionalpdf_data.csv'.
        '''
        
        vmin = kwargs.get('vmin',-25)
        Nres = kwargs.get('Nres', 50)
        plot_figure = kwargs.get('plot_figure', True)
        save_figure = kwargs.get('save_figure', False)
        fig_filename = kwargs.get('fig_filename', 'conditionalpdf_bestfit.png')
        figsize = kwargs.get('figsize', (12,12))
        save_csv = kwargs.get('save_csv', False)
        csv_filename = kwargs.get('csv_filename', 'conditionalpdf_data.csv')
        
        # check if prior steps were done
        if not hasattr(self, 'margLH'):
            print('--------------------------------------------------------')
            print('Approximation of marginal likelihood has not been done.')
            print('Marginal likelihood is automatically assumed as 1e-20')
            self.margLH = 1e-20
            print('--------------------------------------------------------')
        if not hasattr(self, 'cmaes'):
            raise ValueError('Genetic alorithm to find the optimum should be done in prior!')
        
        ## very specific for my problem ##
        best_val_mod = np.delete(self.cmaes['best_ln'].ravel(),[2,5])
        if self.plot_true_values:
            true_param = np.delete(self.true_param_val,[2,5])
        ##################################
        
        fig, axes = plt.subplots(nrows=self.n_var, ncols=self.n_var, figsize=figsize)
        best_val = utils.exp_trans(best_val_mod, self.n_var, self.ln_idx)
        best_val = utils.log10_trans(best_val, self.n_var, self.ln_idx)
        if self.plot_true_values:
            True_paramsList = utils.log10_trans(true_param, self.n_var, self.ln_idx)
        for comb in list(itertools.combinations(range(self.n_var), 2)):
            d1, d2 = comb[0], comb[1] 
            
            par_ax = np.linspace(self.lb_mod[d1], self.ub_mod[d1], Nres)
            par_ay = np.linspace(self.lb_mod[d2], self.ub_mod[d2], Nres)
            # put best value in par_ax and sort
            par_ax = np.sort(np.append(par_ax,best_val_mod[d1]))
            par_ay = np.sort(np.append(par_ay,best_val_mod[d2]))
            par_mat_mod = np.tile(best_val_mod, ((Nres + 1) ** 2, 1))
            for idx, (val_x, val_y) in enumerate(itertools.product(par_ax, par_ay)):
                par_mat_mod[idx,d1] = val_x
                par_mat_mod[idx,d2] = val_y
            logLH = self.calc_LH_NN(par_mat_mod)
            
            post_aa = logLH - np.log(self.margLH)
            post_aa = post_aa.reshape(Nres+1,Nres+1)
            LHS = post_aa.copy() 
            LHS[LHS<vmin]=vmin
            vmin = vmin
            vmax = int(np.max(post_aa) + 1)
            
            if d1>=self.ln_idx:
                par_ax = np.exp(par_ax)
                par_ax = np.log10(par_ax)
            if d2>=self.ln_idx:
                par_ay = np.exp(par_ay)
                par_ay = np.log10(par_ay) 
            axes[d2][d1].contourf(par_ax,par_ay,LHS.T,levels=50,vmin=vmin, vmax=vmax)
            axes[d1][d2].axis('off') # remove the upper triangle
            
            if self.plot_true_values: 
                axes[d2][d1].plot(True_paramsList[d1],True_paramsList[d2],'*',color='C3',markersize=10)
            # plot the best fit values
            axes[d2][d1].plot(best_val[d1],best_val[d2],'C2X',markersize=10)
            if d1==0:
                axes[d2][d1].set_ylabel(label[d2])
            else:
                axes[d2][d1].tick_params(axis='y',labelleft=False,which='both') # remove the ticks
            if d2==self.n_var-1:
                axes[d2][d1].set_xlabel(label[d1])
                axes[d2][d1].tick_params(axis='x', labelrotation = 45, which='both')
            else:
                axes[d2][d1].tick_params(axis='x',labelbottom=False,which='both') # remove the ticks
            
            # save .csv files    
            if save_csv:
                if header_csv.shape[1] == self.n_var:
                    header_csv = header_csv.T
                param1_label = header_csv[d1,0]
                param2_label = header_csv[d2,0]
                param1_unit = header_csv[d1,1]
                param2_unit = header_csv[d2,1]
                
                header_label = [param1_label] + [param2_label]*(Nres+1)
                header_unit = [param1_unit] + [param2_unit]*(Nres+1)
                header_comment = ['']*(Nres+2)
                
                if d1>=self.ln_idx:
                    par_ax = utils.power(10,par_ax)
                if d2>=self.ln_idx:
                    par_ay = utils.power(10,par_ay)
                
                par_ax_save = par_ax.reshape(-1,1)
                par_ay_save = np.insert(par_ay, 0, np.NaN)
                all2D = pd.DataFrame([header_label, header_unit, header_comment, par_ay_save])
                data2D = pd.DataFrame(np.hstack(( par_ax_save,np.log10(np.exp(post_aa)) )))
                all2D = pd.concat([all2D,data2D])
                
                index = str(csv_filename).find('.csv')
                csv_filename_temp = Path(str(csv_filename)[:index] + f'_2D_{d1}{d2}.csv')     # if filename is xxx.csv, convert to i.e. xxx_2D_01.csv
                all2D.to_csv(csv_filename_temp, header = False, index = False)
                
        for comb in range(self.n_var):    
            par_ax = np.linspace(self.lb_mod[comb],self.ub_mod[comb],Nres)
            # put best value in par_ax and sort
            par_ax = np.sort(np.append(par_ax,best_val_mod[comb]))
            par_mat_mod = np.tile(best_val_mod,Nres+1).reshape(Nres+1,self.n_var)
            par_mat_mod[:,comb] = par_ax
            logLH = self.calc_LH_NN(par_mat_mod)

            post_aa = logLH - np.log(self.margLH)
            LHS = post_aa.copy()
            LHS[LHS<vmin]=vmin
            
            vmin = vmin
            vmax = int(np.max(post_aa) + 1)
            
            if comb>=self.ln_idx:
                par_ax = np.exp(par_ax)
                par_ax = np.log10(par_ax)
                
            axes[comb][comb].clear() # clear the previous plot
            axes[comb][comb].plot(par_ax,LHS)

            # plot true values as a line
            if self.plot_true_values:
                axes[comb][comb].axvline(True_paramsList[comb],color='C3',linestyle='--')
                axes[comb][comb].plot(True_paramsList[comb],1,'C3*',markersize=10)
            if comb==0:
                # axes[comb][comb].set_yscale('log')
                axes[comb][comb].set_ylabel('ln P(w|y)') 
                # axes[comb][comb].set_ylim((10**vmin,2)) 
                axes[comb][comb].set_yticks([vmin,int(vmin/2),vmax], minor=False)
                
            else:
                axes[comb][comb].tick_params(axis='y',labelleft=False,which='both')
                axes[comb][comb].tick_params(axis='y',labelright=True,which='major')
                # axes[comb][comb].set_yscale('log')
                # axes[comb][comb].set_ylim((10**vmin,2))
                axes[comb][comb].set_yticks([vmin,int(vmin/2),vmax], minor=False)
            # axes[comb][comb].set_xlim((lb_log10[comb],ub_log10[comb]))
            if comb==self.n_var-1:
                axes[comb][comb].set_xlabel(label[comb])
                axes[comb][comb].tick_params(axis='x', labelrotation = 45, which='both') 
            else:
                axes[comb][comb].tick_params(axis='x',labelbottom=False,which='both') 
                
            if save_csv:
                if header_csv.shape[1] == self.n_var:
                    header_csv = header_csv.T
                param1_label = header_csv[comb,0]
                param1_unit = header_csv[comb,1]
                
                header_label = [param1_label] + ['log Probability \i(P)']
                header_unit = [param1_unit] + ['']
                header_comment = ['','']
                par_ax_save = par_ax.reshape(-1,1)
                if comb>=self.ln_idx:
                    par_ax_save = utils.power(10,par_ax_save)
                all1D = pd.DataFrame([header_label, header_unit,header_comment])
                data1D = pd.DataFrame(np.hstack((par_ax_save,( np.log10(np.exp(post_aa)) ).reshape(-1,1))))
                all1D = pd.concat([all1D,data1D])
                
                index = str(csv_filename).find('.csv')
                csv_filename_temp = Path(str(csv_filename)[:index] + f'_1D_{comb}.csv')     # if filename is xxx.csv, convert to i.e. xxx_2D_01.csv
                all1D.to_csv(csv_filename_temp, header = False, index = False)
        ## Make colorbar
        # Define the logarithmic space for the colorbar
        cmap = plt.get_cmap('viridis')
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        ticks = [vmin,(int(vmin/2)),vmax]
        # Create a scalar mappable to map the values to colors
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.75, 0.6, 0.05, 0.2])  # left, bottom, width, height
        cbar = plt.colorbar(sm, cax=cax, ticks=ticks)
        # Set the colorbar label on the left side
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel('ln P(w|y)', rotation=90, va='bottom')
        
        if save_figure:
            fig.savefig(fig_filename)
            
        if plot_figure:
            fig.show()
        else:
            plt.close()
    '''
    check JV_NN(bestfit)
    '''
    def checkNNfit(self, **kwargs):
        plot_figure = kwargs.get('plot_figure', True)
        save_figure = kwargs.get('save_figure', False)
        fig_filename = kwargs.get('fig_filename', 'test')
        save_csv = kwargs.get('save_csv', False)
        csv_filename = kwargs.get('csv_filename', 'JVASA_bestfit.csv')
        specifier = kwargs.get('specifier', 'test')
        
        curdir = Path().absolute()  # get current directory
        input_path = curdir / 'example_NNmodel' / 'inputs_for_training_data.jsonc'
        with open(input_path) as f: # open input file
            input = jstyleson.load(f) # read input file
        
        if self.plot_true_values:
            keyword = 'sim'
        else:
            keyword = 'exp'
        
        # generate JNN with best fit val      
        y_NN_1 = (self.reg1).predict(self.cmaes["best_norm"], verbose=0)
        y_NN_2 = (self.reg2).predict(self.cmaes["best_norm"], verbose=0)
        y_NN_1 = utils.scale_and_exponentiate(y_NN_1.ravel(), self.y1_min, self.y1_max) + self.x_exp/self.Rpdark[1] + self.x_exp**2/self.Rpdark[2]
        y_NN_2 = utils.scale_and_exponentiate(y_NN_2.ravel(), self.y2_min, self.y2_max)
        # shift back
        y_NN = utils.transform_JV_to_original(np.log(y_NN_1), np.log(y_NN_2), self.NNpoints, self.oneJVlen)
        JV_NNfit = np.vstack((self.x_exp,y_NN)).T
            
        # plot light jv
        fig,ax = plt.subplots(1,2, figsize=[8,6])
        ax[0].semilogy(np.exp(self.y_exp_1))
        ax[0].semilogy(y_NN_1, '--')
        ax[1].semilogy(np.exp(self.y_exp_2))
        ax[1].semilogy(y_NN_2, '--')
        ax[0].legend([f'J {keyword}', 'NN fit'])
        fig.tight_layout()
        if save_figure:
            fig_filename_temp = Path(str(fig_filename) + '_NNfit_light')   
            fig.savefig(fig_filename_temp)
        if not plot_figure:
            plt.close('all')
            
        JV_exp = np.vstack((self.x_exp,self.y_exp)).T
        if save_csv:
            for aa, experiment in enumerate(input["experiments"]):
                index = str(csv_filename).find('.csv')
                led_name = experiment.split('.gen')[0].split('_')[-1]
                csv_filename_temp = Path(str(csv_filename)[:index] + f'_exp_{specifier}_{led_name}.csv')     # if filename is xxx.csv, convert to i.e. xxx_exp_27mA.csv
                JV_now = JV_exp[aa*self.oneJVlen:(aa+1)*self.oneJVlen,:]
                # save the input JV file which was analyzed
                Imf.saveJV2csv(JV_now,f'exp_{specifier}',csv_filename_temp)
                
                csv_filename_temp = Path(str(csv_filename)[:index] + f'_NNfit_{specifier}_{led_name}.csv')     # if filename is xxx.csv, convert to i.e. xxx_exp_27mA.csv
                JV_now = JV_NNfit[aa*self.oneJVlen:(aa+1)*self.oneJVlen,:]
                # save the input JV file which was analyzed
                Imf.saveJV2csv(JV_now,f'NNfit_{specifier}',csv_filename_temp)
        