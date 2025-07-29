from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.factory import get_termination, get_algorithm
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.core.callback import Callback
import numpy

def run_cmaes_reduced_jv(lb, ub, n_var, y_exp, x_exp, sigma_J_phi, 
                         y1_max, y1_min, y2_max, y2_min, reg1, reg2, scaler, obs_mask, Rpdark, verbose=False):
    
    '''
    run_cmaes : optimizer to find a good point
           
    Parameters 
    ----------
    lb (list or array) : lower bound in ln transform form. 
    ub (list or array) : upper bound in ln transform form. 
    n_var (int) : number of variables
    y_exp (array) : J (current density) data in original form (consist of 4 JV curves)
    x_exp (array) : V (voltage) data
    simga_J_phi : uncertainty of measurement
    y1_max, y1_min, y2_max, y2_min : min max values of the first and the second dataset
    reg1 : regression model trained with the first type of data 
    reg2 : regression model trained with the second type of data 
    scaler : scaler used for the transformation of the input parameters for the neural network
    obs_mask (bool) : boolean array which indicates which JV curve(s) is(are) the interest of evaluation

    Returns
    -------
    res.X : parameters predicted res.X in pymoo space, which is from 0 to 1. 
    best_ln : predicted parameters in logarithm space
    best_norm : predicted parameters in min max scaler transformed space
    res.F : error corresponding to the best fit
    callbackdata : sampled points during exploration
    
    '''

    # this_termination = get_termination("n_eval", 10000) # define criterion for the termination of the optimization: maximum number of generations
    
    popsize =200  # population size of the genetic algorithm
    
    # define the optimazition problem as a class of the r
    class MyProblem(Problem):

        def __init__(self, x_exp, y_exp, sigma_J_phi, reg1, reg2, scaler, y1_max, y1_min, y2_max, y2_min, obs_mask, Rpdark):
            super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=numpy.zeros(n_var), xu=numpy.ones(n_var), elementwise_evaluation=False)
            self.scaler = scaler
            self.x_exp= x_exp
            self.y_exp = y_exp
            self.reg1 = reg1
            self.reg2 = reg2
            self.y1_max = y1_max
            self.y1_min = y1_min
            self.y2_max = y2_max
            self.y2_min = y2_min
            self.Rpdark = Rpdark
            self.sigma_J_phi = sigma_J_phi
            self.obs_mask = obs_mask
        def _evaluate(self, theta, out, *args, **kwargs):            
            # calculate the error for a specific set of parameters suggested by the optimization algorithm
            theta_trans = numpy.array(theta) * (ub-lb) + lb 
            error = eval_func_cmaes(self.reg1, self.reg2, self.scaler, self.y1_max, self.y1_min, self.y2_max, self.y2_min,
                                      theta_trans, self.x_exp, self.y_exp, self.sigma_J_phi, self.obs_mask, self.Rpdark)
            out["F"] = error
    
    # Define a custom callback to track the population
    class MyCallback(Callback):
        def __init__(self):
            super().__init__()
            self.data["offspring"] = []

        def notify(self, algorithm):
            # Store all offsprings and their objective values
            self.data["offspring"].append(algorithm.pop.get("X"))

    # define algorithm. 
    algorithm = get_algorithm("cmaes",
                                sigma=0.2,
                                restarts=0,
                                restart_from_best=False,
                                incpopsize=1,
                                popsize=popsize)

    problem = MyProblem(x_exp, y_exp,sigma_J_phi, reg1, reg2, scaler, 
                     y1_max,y1_min,y2_max,y2_min,obs_mask,Rpdark)  # define an object to optimize by the algorithm

    res = minimize(problem, algorithm,('n_gen', 100),callback=MyCallback(),verbose=False)
    best_ln = (res.X * (ub-lb) + lb).reshape(1,-1)                     # best parameters from fit in ln space
    
    ### very specific to my problem ###
    real_fixed_value = 1.3
    best_ln = numpy.insert(best_ln, 2, real_fixed_value, axis=1)
    best_ln = numpy.insert(best_ln, 5, best_ln[0,4],axis=1)
    
    best_norm = scaler.transform(best_ln)               # best parameters from fit in scaler transformed space
    if verbose:
        print("cmaes calculation took --- %s seconds ---" % (res.exec_time))
        print(f"Best point at {best_ln} with score {res.F}")
    callback_data = res.algorithm.callback.data
    return res.X, best_ln, best_norm, res.F, callback_data

def eval_func_cmaes(reg1,reg2,stscaler,y1_max,y1_min,y2_max,y2_min,theta,x_exp,y_exp,sigma_J_phi,reg2_mask,Rpdark,NNpoints=512,oneJVlen=128):
    
    below_Voc_mask = y_exp<0
    theta = numpy.insert(theta,2,1.3,axis=-1)
    theta = numpy.insert(theta,5,theta[:,4],axis=-1)
    
    theta_norm = stscaler.transform(theta)
    y1 = reg1.predict(theta_norm ,verbose=0)
    y2 = reg2.predict(theta_norm ,verbose=0)
    
    y1_scaled = numpy.array(y1*(y1_max-y1_min)+y1_min)
    y2_scaled = numpy.array(y2*(y2_max-y2_min)+y2_min)
    y1_scaled = numpy.exp(y1_scaled)     # mA/cm²
    y2_scaled = numpy.exp(y2_scaled)     # mA/cm²
    # transformed in a original scale 
    for ii in range(NNpoints//oneJVlen):
        y1_scaled[:,ii*oneJVlen:(ii+1)*oneJVlen] = y1_scaled[:,ii*oneJVlen:(ii+1)*oneJVlen] - y2_scaled[:,ii].reshape((y2_scaled.shape[0],1))
    y1_scaled += x_exp/Rpdark[1] + x_exp**2/Rpdark[2] 
    
    errors = list()
    for ii in range(NNpoints//oneJVlen):
        phi_mask = numpy.repeat([False],NNpoints)
        phi_mask[ii*oneJVlen:(ii+1)*oneJVlen] = below_Voc_mask[ii*oneJVlen:(ii+1)*oneJVlen]
        error = (y1_scaled[:,phi_mask]- y_exp[phi_mask])
        error = numpy.sqrt(numpy.mean(error**2, axis = -1))
        errors.append(error)

    errors = numpy.vstack(errors).T 
    approx_post = numpy.sum(errors[:,reg2_mask]/sigma_J_phi[reg2_mask],axis=-1)
    return approx_post

def run_cmaes_reduced_jv_shifted(lb, ub, n_var, y_exp, y1_exp, y2_exp, x_exp, sigma_J_phi, 
                         y1_max, y1_min, y2_max, y2_min,
                         reg1, reg2, scaler, obs_mask, Rpdark, width, verbose=False):
    
    '''
    run_cmaes : optimizer to find a good point
           
    Parameters 
    ----------
    lb (list or array) : lower bound in ln transform form. 
    ub (list or array) : upper bound in ln transform form. 
    n_var (int) : number of variables
    y_exp (array) : J (current density) data in original form (consist of 4 JV curves)
    x_exp (array) : V (voltage) data
    simga_J_phi : uncertainty of measurement
    y1_max, y1_min, y2_max, y2_min : min max values of the first and the second dataset
    reg1 : regression model trained with the first type of data 
    reg2 : regression model trained with the second type of data 
    scaler : scaler used for the transformation of the input parameters for the neural network
    obs_mask (bool) : boolean array which indicates which JV curve(s) is(are) the interest of evaluation

    Returns
    -------
    res.X : parameters predicted res.X in pymoo space, which is from 0 to 1. 
    best_ln : predicted parameters in logarithm space
    best_norm : predicted parameters in min max scaler transformed space
    res.F : error corresponding to the best fit
    callbackdata : sampled points during exploration
    
    '''

    # this_termination = get_termination("n_eval", 10000) # define criterion for the termination of the optimization: maximum number of generations
    
    popsize =200  # population size of the genetic algorithm
    
    # define the optimazition problem as a class of the r
    class MyProblem(Problem):

        def __init__(self, x_exp, y_exp, y1_exp, y2_exp, sigma_J_phi, reg1, reg2, scaler, 
                     y1_max, y1_min, y2_max, y2_min, obs_mask, Rpdark, width):
            super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=numpy.zeros(n_var), xu=numpy.ones(n_var), elementwise_evaluation=False)
            self.scaler = scaler
            self.x_exp= x_exp
            self.y_exp = y_exp
            self.y1_exp = y1_exp
            self.y2_exp = y2_exp
            self.reg1 = reg1
            self.reg2 = reg2
            self.y1_max = y1_max
            self.y1_min = y1_min
            self.y2_max = y2_max
            self.y2_min = y2_min
            self.Rpdark = Rpdark
            self.sigma_J_phi = sigma_J_phi
            self.obs_mask = obs_mask
            self.width = width
        def _evaluate(self, theta, out, *args, **kwargs):            
            # calculate the error for a specific set of parameters suggested by the optimization algorithm
            theta_trans = numpy.array(theta) * (ub-lb) + lb 
            error = eval_func_cmaes_shifted(self.reg1, self.reg2, self.scaler,
                                    self.y1_max, self.y1_min, self.y2_max, self.y2_min,
                                    theta_trans, self.x_exp, self.y_exp, self.y1_exp, self.y2_exp, self.sigma_J_phi, self.obs_mask, self.Rpdark, self.width)
            # error = eval_func_cmaes_stdscaled(self.reg1, self.reg2, self.scaler,
            #                         self.y1_max, self.y1_min,
            #                         theta_trans, self.x_exp, self.y_exp, self.y1_exp, self.y2_exp, self.obs_mask, self.Rpdark, self.width)
            out["F"] = error
    
    # Define a custom callback to track the population
    class MyCallback(Callback):
        def __init__(self):
            super().__init__()
            self.data["offspring"] = []

        def notify(self, algorithm):
            # Store all offsprings and their objective values
            self.data["offspring"].append(algorithm.pop.get("X"))

    # define algorithm. 
    algorithm = get_algorithm("cmaes",
                                sigma=0.2,
                                restarts=0,
                                restart_from_best=False,
                                incpopsize=1,
                                popsize=popsize)

    problem = MyProblem(x_exp, y_exp,y1_exp, y2_exp, sigma_J_phi, reg1, reg2, scaler,
                     y1_max,y1_min,y2_max,y2_min,obs_mask,Rpdark, width)  # define an object to optimize by the algorithm

    res = minimize(problem, algorithm,('n_gen', 100),callback=MyCallback(),verbose=False)
    best_ln = (res.X * (ub-lb) + lb).reshape(1,-1)                     # best parameters from fit in ln space
    
    ### very specific to my problem ###
    real_fixed_value = 1.3
    best_ln = numpy.insert(best_ln, 2, real_fixed_value, axis=1)
    best_ln = numpy.insert(best_ln, 5, best_ln[0,4],axis=1)
    
    best_norm = scaler.transform(best_ln)               # best parameters from fit in scaler transformed space
    if verbose:
        print("cmaes calculation took --- %s seconds ---" % (res.exec_time))
        print(f"Best point at {best_ln} with score {res.F}")
    callback_data = res.algorithm.callback.data
    return res.X, best_ln, best_norm, res.F, callback_data

def eval_func_cmaes_shifted(reg1,reg2,stscaler,y1_max,y1_min,y2_max,y2_min,
                    theta,x_exp,y_exp, y1_exp, y2_exp, sigma_J_phi,reg2_mask,Rpdark,width,NNpoints=512,oneJVlen=128):
  
    # combine below_Voc_mask and reg2_mask
    below_Voc_mask = y_exp<0
    
    y_exp = y_exp.ravel()
    
    theta = numpy.insert(theta,2,1.3,axis=-1)
    theta = numpy.insert(theta,5,theta[:,4],axis=-1)
    
    theta_norm = stscaler.transform(theta)

    # Rpdark_temp = numpy.repeat(Rpdark[1],oneJVlen*4) # extend Rpdark into the length of NN model
    y1 = reg1.predict(theta_norm ,verbose=0)
    y2 = reg2.predict(theta_norm ,verbose=0)
    
    # including Rpdark current
    y1_scaled = scale_and_exponentiate(y1, y1_min, y1_max)      # A/m²
    y2_scaled = scale_and_exponentiate(y2, y2_min, y2_max)      # A/m²
    y1_scaled += x_exp/Rpdark[1] + x_exp**2/Rpdark[2] 
    sigma_exp = numpy.gradient(y1_exp, x_exp)
    
    error_y1 = list()
    for ii in range(NNpoints//oneJVlen):
        phi_mask = numpy.repeat([False],NNpoints)
        phi_mask[ii*oneJVlen:(ii+1)*oneJVlen] = True
        error = (y1_scaled[:,phi_mask]- y1_exp[phi_mask]) / sigma_exp[phi_mask]
        error = numpy.sqrt(numpy.mean(error**2, axis = -1))
        error_y1.append(error)

    error_y1 = numpy.vstack(error_y1).T
    error_y1 = numpy.sum(error_y1[:,reg2_mask], axis=-1)
    error_y2 = (y2_scaled- y2_exp) / sigma_J_phi
    error_y2 = numpy.sqrt(numpy.mean(error_y2[:, reg2_mask]**2,axis=-1))
    penalty = (numpy.tanh(width*error_y2-numpy.pi)-numpy.tanh(width*error_y2+numpy.pi))*10+21
    error_y12 = error_y1 * numpy.transpose(penalty)

    return error_y12

def eval_func_cmaes_stdscaled(reg1,reg2,stscaler,y1_max,y1_min,
                    theta,x_exp,y_exp, y1_exp, y2_exp,reg2_mask,Rpdark,width,oneJVlen=128):
    
    # # cut out the range where J>0
    # below_Voc_mask = y_exp<0
    reg2_mask_temp = numpy.repeat(reg2_mask, oneJVlen)
    # below_Voc_mask = numpy.logical_and(below_Voc_mask, reg2_mask_temp)
    
    theta = numpy.insert(theta,2,1.3,axis=-1)
    theta = numpy.insert(theta,5,theta[:,4],axis=-1)
    
    theta_norm = stscaler.transform(theta)   

    # Rpdark_temp = numpy.repeat(Rpdark[1],oneJVlen*4) # extend Rpdark into the length of NN model
    y1 = reg1.predict(theta_norm ,verbose=0)
    y2 = reg2.predict(theta_norm ,verbose=0)
    
    # including Rpdark current
    y1_scaled = scale_and_exponentiate(y1, y1_min, y1_max)      # A/m²
    y1_scaled += x_exp/Rpdark[1] + x_exp**2/Rpdark[2] 
    # scale back
    y1_scaled = log_and_standardize(y1_scaled, y1_min, y1_max)
    
    error_y1 = numpy.sqrt(numpy.mean((y1_scaled[:,reg2_mask_temp] - y1_exp[reg2_mask_temp])**2, axis=-1))
    error_y2 = numpy.sqrt(numpy.mean((y2[:,reg2_mask] - y2_exp[reg2_mask])**2, axis=-1))
    penalty = (numpy.tanh(width*error_y2-numpy.pi)-numpy.tanh(width*error_y2+numpy.pi))*10+21
    
    return error_y1*numpy.transpose(penalty)

def scale_and_exponentiate(pred, min_val, max_val):
    return numpy.exp(pred * (max_val - min_val) + min_val)

def log_and_standardize(pred, min_val, max_val): 
    return (numpy.log(pred) - min_val) / (max_val - min_val)