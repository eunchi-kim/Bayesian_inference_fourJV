import numpy
from pathlib import Path
import h5
import os
import subprocess
import time



def calculated_inputs(old_input):
      '''
      calculated_inputs : calculate certain fields in the input dictionary to match the ASA template

      here, different calculations can be performed, see uncommented examples
                  
      Parameters 
      ----------
      old_input : dictionary of input data 
      
      returns
      -------
      new_input[XYZ] : manipulated elements of dictionary
      '''  
      new_input = old_input # copy old input dictionary
      # define the type of the deep defect
      if old_input["pol"]:
            new_input["eneg"] = -old_input["eg_abs1"]/2
            new_input["eneut"] = -3
      else:
            new_input["eneg"] = 3
            new_input["eneut"] = -old_input["eg_abs1"]/2

      return new_input["eneg"], new_input["eneut"]
  

def asa(input, template_path, cas_path, cur_dir):
      '''
      asa : write cas file and execute ASA
                  
      Parameters 
      ----------
      input : dictionary of input data
      template_path : filepath of the template file for the cas
      cas_path : filepath of the cas file that is rewritten
      cur_dir : current directory of data_gen.py
      
      returns
      -------
      jv_illum : array of current-density voltage data [V,A/m²]
      '''      
      template = template_path.read_text() # load text from cas template file
      asa_data = template.format(**input) # replace place holders with data from dictionary
      with cas_path.open('w') as file:    # open cas file to write
         file.write(asa_data) # print template text varied with the inputs into cas file
      subprocess.run([cur_dir + "ASA\\asa5.exe", "-f", cas_path._str], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # run cas file and supress the output by ASA
      jv_illum = numpy.loadtxt('.\\' + cur_dir + "ASA/jv_illum.dat")    # load the result of the ASA simulation [V, A/m²]
      os.remove('.\\' + cur_dir + "ASA/jv_illum.dat") # delete the dat result file
      return jv_illum
         

def train_data_gen(par_mat, input, seq, template_path, data, cur_dir):
      '''
      train_data_gen : prepare simulations, run them and process results
                  
      Parameters 
      ----------
      par_mat : dictionary of input data where strings are saved as bytes (?)
      
      returns
      -------
      decoded_input : dictionary with all strings in normal style
      '''      
      params = input['var_param']   # all parameters that are supposed to be varied
      calculated_pars = numpy.zeros((seq,2))    # preallocate parameters that have to be changed to fit ASA template (size might change according to variables)
      cas_path = Path("%sASA/%s.cas" % (cur_dir, input["fl_cas"]))      # define filepath for actual cas file
      
      data.root.vol_swp = numpy.round(numpy.arange(input["vstart"], (input["vend"] + input["vstep"]), input["vstep"]), decimals = 2) # define voltage sweep from input parameters
      jv_sim = numpy.zeros((seq,len(data.root.vol_swp))) # preallocate current-density voltage array
      start_time = time.time()   # track start time for run time estimation
      for idx in range(seq):  # loop though entire sequence
         for var in range(input['n_var']):      # loop through variable parameters
               input[params[var]] = par_mat[idx,var]  # replace value for current variable with value from parameter matrix
         calculated_pars[idx,:]=calculated_inputs(input) # manipulate input for this simulation run
         jv_illum =  asa(input, template_path, cas_path, cur_dir) # run ASA
         jv_sim[idx,:] = jv_illum[:,1].T  # add transposed JV curve to collection of JV curves [V, A/m²]
      #    if idx%1000 == 0 or idx == seq-1:      
      #       data.root.jv_sim = jv_sim
      #       data.save()
      #       print(str(idx) + ":--- " + str(time.time() - start_time) + " seconds ---")
      data.root.jv_sim = jv_sim     # write JV data into h5 object [V, A/m²]
      data.root.calculate_pars = calculated_pars
      data.save() # save to hdf5 file
      print("--- %s seconds ---" % (time.time() - start_time))
      

def decode_dictionary(encoded_input):
      '''
      decode_dictionary : decode byte style data from h5 object
                  
      Parameters 
      ----------
      encoded_input : dictionary of input data where strings are saved as bytes (?)
      
      returns
      -------
      decoded_input : dictionary with all strings in normal style
      '''
      decoded_input = encoded_input # copy enncoded input
      for category in encoded_input:      # loop through every field in dictionary
            if encoded_input[category].size==1: # test if content of field is single input
                  try:
                        decoded_input[category] = encoded_input[category].decode('utf-8') # decode field
                  except:
                        decoded_input[category] = encoded_input[category] # if it fails, just copy field without decoding (e.g. if content is double)
            else: # in case content is list
                  yy = list() # preallocate list for decoded content
                  for xx in encoded_input[category]:  # loop through list
                        try:
                              xx = xx.decode('utf-8') # decode element of list
                        except:
                              xx = xx     # if decoding failed, just use undecoded element
                        yy.append(xx)     # append decoded element to list
                  decoded_input[category] = yy  # define decoded list as element for this category

      return decoded_input
      
    

def main(FN_seq, identifier, seq_no,cpu_num):
      '''
      data_gen : run simulation to generate training data
                  
      Parameters 
      ----------
      FN_seq : filename of input hdf5 file
      identifier : string with date and time for unique filenames
      seq_no : number of subset of the parameter sequence
      '''

      data = h5.H5()    # define h5 object to store the simulated data
      cur_dir = identifier + '\\' + seq_no + '\\'     # define directory of the data_gen.py file
      data.filename = Path(cur_dir + FN_seq)    # define filepath where the input file is located
      data.load() # load input h5 file
      input = data.root.input # extract input dictionary from h5 object
      input = decode_dictionary(input)      # decode elements of dictionary
      par_mat = data.root.par_mat   # read parameter matrix from input file
      batchsize = int(input['seq']/cpu_num)     # calculate size of subset
      template = Path(cur_dir + 'ASA/' + input["fl_template"]) # define filepath of template for cas file
      input['seq'] = batchsize      # change size of sequence in input dictionary
      input['dir'] = cur_dir  # change directory to subfolder of data_gen.py in input dictionary
      train_data_gen(par_mat, input, input['seq'], template, data, cur_dir)   # run training data simulations
      data.save() # save results to h5 file


   



if  __name__ == "__main__":
      main() 


