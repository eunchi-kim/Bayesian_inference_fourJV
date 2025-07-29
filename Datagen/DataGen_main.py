# %% import  packages and selfwritten scripts 
import gen_seq  # generates sobol sequence
import h5   # handles hdf5 file format
import os # here: get current directory
import shutil   # copy files and folders
import multiprocessing  # run multiple calculations in parallel
import numpy    #  maths and array handeling
import pathlib  # handle paths independent of operation system
import jstyleson
import sys

# %% execution funciton
def run_datagen(dir, FN_seq, counter,cpu_num):
    ''' 
    run_datagen: imports and runs data generation scripts in different folders. There is probably a prettier way to do this than just import the data_gen script each time with a different name.

    Parameters:
    -------------------
    dir: directory, where the data_gen.py is located relative to the current folder
    FN_seq: filename of the sequence of parameters that is supposed to be simulated
    counter: number of executions of this function
    '''
    identifier, seq_no = dir.split('\\')[-2:]   # splits directory into its subfolders
    exec('import %s.%s.data_gen as data_gen%g' %(identifier, seq_no, counter))  # executes command that import of data_gen.py file as unique (guaranteed by counter)
    exec('data_gen%g.main(%s, %s, %s, %s)' %(counter, 'FN_seq', 'identifier', 'seq_no','cpu_num')) # executes command to run imported data_gen file
    

    # %% main script
if __name__ == "__main__" : 
    # define how many core processors are necessary
    cpu_num = 2
    ln_idx = 4
    # generate sequence for training data
    FN_seqAll, input = gen_seq.main(cpu_num, ln_idx)
    # with open('inputs_for_training_data.jsonc') as f: # open input file
    #    input = jstyleson.load(f) # read input file
    curdir = os.getcwd()    # find current directory 
    template_dir = curdir + '\\' + input["gen_folder_template"]     # directory of template generation routine folder
    dir_seq_all = numpy.empty((len(input["experiments"]),cpu_num),dtype=object)
    exp_folders = list()   # preallocate list for all directories that specify the experiment (e.g. illumination spectrum or thickness variation) 
    for aa, experiment in enumerate(input["experiments"]):  # loop through all experiment variations
        # create name for the subfolders belonging to each experiment (for now: specific to generation file variation)
        folder_name = experiment.split('.gen')[0]
        folder_name = 'datagen_' + folder_name.split("B23n8-3_")[-1]
        exp_folders.append(curdir + '\\' + folder_name) # add folder for current experiment to list
        if not os.path.exists(exp_folders[aa]):
            os.makedirs(exp_folders[aa])
            for num in range(cpu_num):
                dir_seq_temp = exp_folders[aa] + '\\' + f'seq_{num+1}'
                shutil.copytree(template_dir, dir_seq_temp)  # copy template folder to new experiment folder
                dir_seq_all[aa,num] = dir_seq_temp
                del dir_seq_temp
        else:
            print("folder exists! Change the folder name or remove it")
            sys.exit() 

    counter = 1 # start counter for import of data_gen.py
    for bb, experiment in enumerate(input["experiments"]):  # loop through all experiment variations
        processes = list()
        filenameAll = list()
        for num in range(cpu_num):
            subset_data = h5.H5() # define h5 object for the first sequence of simulations
            subset_data.filename = pathlib.Path(curdir + '\\' + FN_seqAll[num])  # use file that already contains subset of parameter matrix
            subset_data.load() # load input h5 file with parameter matrix subset
            subset_data.root.input[input["experiment_type"]] = experiment # vary field in input directory for this experiment
            subset_data.filename = pathlib.Path(dir_seq_all[bb,num] + '\\' + FN_seqAll[num]) # change filename to experiment folder
            filenameAll.append(dir_seq_all[bb,num] + '\\' + FN_seqAll[num])
            subset_data.save() # save varied input h5 file to specific experiment and subsequence folder
            
            # define multiple simulation processes in parallel
            p = multiprocessing.Process(target=run_datagen, args=(dir_seq_all[bb,num], FN_seqAll[num], counter,cpu_num )) # define first process to run the run_datagen function with the data for the first sequence
            processes.append(p)
            counter +=1 # raise counter to facilitate the import of the next data_gen.py file
        for p in processes:# start all processes
            p.start()
        for p in processes: # wait until all processes are finished
            p.join()

        print('round %s finished' %(bb))
        # clear all processes
        del processes

        jv_sim_merge = list()
        par_mat_merge = list()
        for num, filename in enumerate(filenameAll):
            sub_result = h5.H5()
            sub_result.filename = filename
            sub_result.load()
            jv_sim_merge.append(sub_result.root.jv_sim)      # extract the simulated current-density voltage data [V, A/m²]
            par_mat_merge.append(sub_result.root.par_mat)    # extract subset of parameter matrix for this simulation set
        
        jv_sim_merge = numpy.concatenate(jv_sim_merge, axis=0)      # merge all sets of jv curves
        par_mat_merge = numpy.concatenate(par_mat_merge, axis=0)    # merge all sets of parameter matrices
        
        data_merge = h5.H5()    # define h5 object to save the merged result data
        data_merge.filename = pathlib.Path(exp_folders[bb]  + '\\' + FN_seqAll[0].split('seq_1')[0] + exp_folders[bb].split('\\')[-1] + "_" + str(input["seq"]) + '_merged.h5') # create filename in experiment specific folder
        data_merge.root.Input = sub_result.root.input    # write manipulated input into result h5 object
        data_merge.root.vol_swp = sub_result.root.vol_swp    # write voltage sweep into result h5 object
        data_merge.root.lb = sub_result.root.input["lb"]     # write lower boundary  into result h5 object
        data_merge.root.ub = sub_result.root.input["ub"]     # write upper boundary into result h5 object
        data_merge.root.jv_sim = jv_sim_merge           # write jv curves into result h5 object
        data_merge.root.par_mat = par_mat_merge         # write parameter matrix into result h5 object
        data_merge.root.Input["seq"] = len(jv_sim_merge)    # write full length of sequence into input dictionary
        data_merge.save() # save merged result h5 file

    print('all finished')

