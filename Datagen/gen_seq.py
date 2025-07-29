import numpy
from pathlib import Path
import datetime
import h5
import jstyleson
import sobol_seq


def read_input(input_dir, identifier, data):
    '''
    read_input : read import jsonc file
            
    Parameters 
    ----------
    input_dir : filename of input file
    identifier : string with date and time for unique filenames
    data : h5 variable for data storage

    returns
    -------
    input : directory with all the input data from jsonc file
    '''
    with open(input_dir) as f: # open input file
       input = jstyleson.load(f) # read input file
       data.root.input = input      # write input into h5 file
    data.filename = identifier + '-'+ input['name']+'.h5'   # define filename for input data with specific timestamp
    data.save() # save input data
    return input

def gen_par_seq(lb, ub, seq, params, data, ln_idx):
      '''
      gen_par_seq : generates parameter matrix with sobol seqence for vaiable parameters

      Parameters
      -------
      lb: lower boundary of parameters
      ub: upper boundary of parameters
      seq: length of sequence/ number of parameter combination desired
      data: h5 variable for data storage
                  
      returns
      -------
      par_mat: parameter sobol sequence of size (seq, n_var)
      '''
            
      lb_mod = numpy.concatenate((lb[:ln_idx],numpy.log(lb[ln_idx:])),axis=0)     # take logarithm of lower boundary of parameters that are varied over orders of magintude
      ub_mod = numpy.concatenate((ub[:ln_idx],numpy.log(ub[ln_idx:])),axis=0)     # take logarithm of upper boundary of parameters that are varied over orders of magintude
      # lb_mod = numpy.array(lb)
      # ub_mod = numpy.array(ub)
      
      par_mat_mod = (sobol_seq.i4_sobol_generate(len(params), seq) * (ub_mod - lb_mod) + lb_mod)      # generate sobol sequence from 0 to 1 and scale it to the boundaries
      par_mat = numpy.concatenate((par_mat_mod[:,:ln_idx],numpy.exp(par_mat_mod[:,ln_idx:])),axis=1)      # transform logarithm back to real values
      
      data.root.par_mat_ln = par_mat_mod  # save logarithm version of parameter matrix
      data.root.par_mat = par_mat   # save true version of parameter  matrix
      data.save() # save input and sequence data
      return par_mat
  

def main(cpu_num, ln_idx):
      '''
      gen_seq : generates sobol seqence for vaiable parameters
                  
      returns
      -------
      data1.filename: filename of first third of the seqence
      data2.filename: filename of second third of the seqence
      data3.filename: filename of last third of the seqence
      input: all input data for simulations imoprted from inputs_for_training_data.jsonc
      '''
      
      identifier = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")    # create unique identifier for this simulation run
      input_json = Path('inputs_for_training_data.jsonc')   # define name of input file
      data = h5.H5() # create h5 class variable for the input data
      input = read_input(input_json, identifier, data) # import data from jsonc file
      
      lb = numpy.asarray(input['lb']) # extract lower boundary for parameters from input dictionary
      ub = numpy.asarray(input['ub'])     # extract upper boundary for parameters from input dictionary
      input["n_var"] = lb.size      # extract number of variable parameters from length of lower boundary
      par_mat = gen_par_seq(lb, ub, input['seq'], input['var_param'], data, ln_idx)    # generate parameter matrix with all parameter combinations to be simulated
      data.save() # save input and sequence data
      
      # split parameter sequence into 3 part to be run in parallel
      batchsize = int(input['seq']/cpu_num)     # length of each subset
      data_filename_all = list()
      for num in range(cpu_num):
            subset_data = h5.H5()  # create new h5 object for subset
            data_filename_all.append(f"{identifier}-{input['n_var']}_param_seq_{num+1}.h5")
            subset_data.filename = f"{identifier}-{input['n_var']}_param_seq_{num+1}.h5"  # write filename for subset
            subset_data.root.input = input  # write input dictionary into h5 object

            # determine the range for the parameter matrix subset
            start_idx = num * batchsize
            end_idx = (num + 1) * batchsize if num < cpu_num else None  # ensure the last subset includes any remaining parameters
            subset_data.root.par_mat = par_mat[start_idx:end_idx, :]  # write subset of parameter matrix into h5 object

            subset_data.save()  # save subset data


      return data_filename_all, input



if  __name__ == "__main__":
      main()