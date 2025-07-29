import numpy
from cadet import H5
from pathlib import Path
import os
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense, Lambda,Conv1D,Conv1DTranspose, Conv2DTranspose, LeakyReLU,Activation,Flatten,Reshape
from tensorflow.keras.models import load_model
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
# from tensorflow.keras.layers import Input, Dense,Conv1D,Conv1DTranspose,Activation,Reshape
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil

def create_dir(name, identity):
    '''
    create_dir : creates the folder where the traning results will be stored
            
    Parameters 
    ----------
    name : name of the current training
    identity : unique date-time signature of the training 

    returns
    -------
    dir_path : path to the directory where training results will be stored
    dir_name : root name of all files associated with this simulation
    '''
   
    cur_dir = Path(os.getcwd())/('Results') # get directory of the results folder
    if os.path.exists(cur_dir):     # check if results folder already exists
        print("file already exists")    
    else:
        os.mkdir(cur_dir)   # create folder if it doesn't exist
    dir_name = identity + name  # create combination of training name and timestamp
    dir_path = cur_dir/("%s" % dir_name)    # create path for new folder
    os.mkdir(dir_path)  # make folder specific to this training
    return dir_path , dir_name


def load_dataset(loc):
    '''
    load_dataset : load y, min, max, and par_mat from dataset file for different types of experiments
            
    Parameters 
    ----------
    loc : path to the dataset

    returns
    -------
    y_norm : numpy array of min-max scaled value of y-axis of experiment
    y_max : scalar value. maximum of the y-axis values of experiment
    y_min : scaler value. minimum of the y-axis values of experiment
    par_mat : numpy array of all parameter combinations in their original space. (NOT ln transformed or scaler transformed)
    ub : numpy array of upper bound
    lb : numpy.array of lower bound
    n_var : number of parameters
    '''
    dataset = H5()
    dataset.filename = Path(loc)
    dataset.load()
    y1_norm = dataset.root.Y1_norm
    y1_max = dataset.root.Y1_max
    y1_min = dataset.root.Y1_min
    y2_norm = dataset.root.Y2_norm
    y2_max = dataset.root.Y2_max
    y2_min = dataset.root.Y2_min
    par_mat = dataset.root.par_mat
    ub = dataset.root.Input.ub
    lb = dataset.root.Input.lb
    n_var = ub.shape[0]
    return y1_norm, y2_norm, y1_max, y1_min, y2_max, y2_min, par_mat, ub, lb, n_var
    


def par_fit_transform(par_mat, n_var):
    '''
    par_fit_transform : perform StandardScaler transformation of the input parameters into the space required by the Neural Network. 
                        before aplying StandardScaler transformation a natural log transform is done on all the inputs.
    
    Parameters
    ----------
    par_mat : numpy.array of all parameter combinations in their original form. 
    n_var : number of variable parameters

    Returns
    -------
    par_norm : standardnorm of par_mat
    scaler : scaler value for normalization 
    ''' 

    par_mat_ln = log_trans(par_mat, n_var)   # get the logarithm of those parameters that are varied over several orders of magnitude  
    
    scaler = StandardScaler()   # define scaler object
    par_norm = scaler.fit_transform(par_mat_ln) # transform data with the standard scaler
    return par_norm, scaler



def par_transform(par_mat_new, scaler):
    '''
    par_transform : performs scaler transform on any new combination of parameter using the same scaler generated in par_fit_transform.

    Parameters 
    ----------
    par_mat_new : numpy.array of new parameter combinations in their original form.
    scaler : scaler value for normlization. 

    Returns
    -------
    par_norm_new : standardnorm of par_matnew
    '''
    par_mat_ln = numpy.concatenate((par_mat_new[:,:3],numpy.log(par_mat_new[:,3:])),axis=1)
    par_norm_new = scaler.transform(par_mat_ln)
    return par_norm_new


def plot_sim(y, dir,fname):
    '''
    plot_sim : plot's input vs arbitrary output

    Parameters 
    ----------
    y : numpy.array of y-axis values. Each row is new result.
    dir : directory where it saves the plot
    fname : filename of the saved plot 

    Returns
    -------
    Nothing
    '''
    num = y.shape[0]    # find number of subplots needed

    fig,ax = plt.subplots(num,1, figsize=[10,15])   # define figure
    for i, yslice  in enumerate(y): # loop through the individual simulations
        ax[i,].plot(yslice) # plot current simulation with arbitrary x axis
    plt.xlabel('x (a.u.)')
    plt.ylabel('y (a.u.)')
    # plt.show()
    fig.savefig(dir/fname)  # save figure
    plt.close
    

def network512(x, y, training_folder, training_name, lrate, batchsize):
    '''
    network512 : NN model used for training surrogate model
           
    Parameters 
    ----------
    x : numpy.array of inputs to NN. Material parameters in their standard normalized form(par_norm). Axis = 0 should be equal to axis = 0 of y.
    y : numpy.array of y-axis values. Each row is new result.  Axis = 0 should be equal to axis = 0 of x. axis = 1 must have 512 points
    training folder : folder where the training results will be stored after completion of training.
    training_name : name of hdf file in which the weights and biases will be stored

    Saves
    -------
    reg : regression model
    reg_name : path where the regression model is stored
    x_test : the x values used for testing
    y_test : the y values used for testing
    x_train : the x_values used for training and validation
    y_train : the y values used for training and validation
    '''

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.10, shuffle=False)    # split training data set to keep certain data set unknown to the NN
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # Network Parameters
    max_filter = 256
    strides = [2,2,2,2]
    kernel = [2,2,2,2]
    map_size = 32

    nParams = X_train.shape[1]

    # define layer of the NN
    z_in = tf.keras.layers.Input(shape=(nParams,))
    z1 = tf.keras.layers.Dense(max_filter)(z_in)
    z1 = tf.keras.activations.swish(z1)
    z1 = tf.keras.layers.Dense(max_filter*map_size)(z1) #256 * 32
    z1 = tf.keras.activations.swish(z1)
    z1 = tf.keras.layers.Reshape((map_size,max_filter))(z1) # 32 by 256
    z2 = tf.keras.layers.Conv1DTranspose( max_filter//2, kernel[3], strides=strides[3], padding='SAME')(z1) # 64 by 128
    z2 = tf.keras.activations.swish(z2)
    z3 = tf.keras.layers.Conv1DTranspose(max_filter//4, kernel[2], strides=strides[2],padding='SAME')(z2) # 128 by 64
    z3 = tf.keras.activations.swish(z3)
    z4 = tf.keras.layers.Conv1DTranspose(max_filter//8, kernel[1], strides=strides[1],padding='SAME')(z3) # 256 by 32
    z4 = tf.keras.activations.swish(z4)
    z5 = tf.keras.layers.Conv1DTranspose(1, kernel[0], strides=strides[0],padding='SAME')(z4) # 512 by 1
    decoded_Y = tf.keras.activations.swish(z5)
    decoded_Y = tf.keras.layers.Reshape((Y_train.shape[1],))(decoded_Y)
    
    # in case the learning rate is not constant, define the scheduler here
    def scheduler(epoch, lr):
        if epoch <250:  # number of epoches, where the first learning rate is valid
            lr = 0.001
            return lr
        elif ((epoch>=250) & (lr>lrate)) or ((epoch>=2000) & (lr>0.0001)) :
            lr = lr*0.99 # decrease learning rate with every step
            return lr        
        else  :
            return lr
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)   # create the tensorflow object for the scheduler 
      
    log_folder = training_folder /  training_name # define folder where data for tensorboard and checkpoint are stored
    print(log_folder)   # print folder name so it can be copied for tensorboard
    tf_callbacks = [TensorBoard(log_dir=log_folder,     # allow tensorflow to store information for tensorboard
                        histogram_freq=1,
                        update_freq='epoch',
                         profile_batch=(2,10))]
    # to activate tensorboard to observe the training losses, activate the right environment in conda and use the command 'tensorboard --logdir log_folder'

    # currently, checkpoints are saved after each iteration, I might uncomment this. To load the checkpoint, use 'reg.load_weights(checkpoint_path)'
    # checkpoint_path = log_folder / ("cp.ckpt")  # define paths where the checkpoints are stored
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,  verbose=1)  # define how checkpoints are made

    reg = Model(z_in,decoded_Y) # create a model with the layers defined before
    reg.summary()   # print information on model
    
    # the neural network can stop before the full number of epoches are run. Here, the criterion for this is the validation loss. 
    # If it has not varied by the min_delta, it will run another 'patience' epochs before terminating the training. Then the best 
    # weights will be used based on the mode.
    es = EarlyStopping(monitor='val_loss', min_delta=5e-10, mode='min', verbose=1, patience=200, restore_best_weights=True) 
    
    reg.compile(loss='mse',optimizer='adam', metrics=['mae', 'msle'])   # mean square error is used for training with the optimizer 'adam'

    # run the training of the neural network.
    reg.fit(X_train,Y_train,shuffle=False, batch_size=batchsize, epochs = 100,
            validation_split=0.4,  callbacks=[lr_callback, tf_callbacks], verbose = 0)
        
    reg_name = Path(training_folder)/("%s_trained_model.h5" % training_name)    # create path to store the neural network
    # save the neural network
    reg.save(reg_name)  
    reg.save(Path(training_folder)/("model" ))
    return reg, reg_name, X_test, Y_test, X_train, Y_train

def network4(x, y, training_folder, training_name, lrate, batchsize):
    '''
    network4 : NN model used for training surrogate model
           
    Parameters 
    ----------
    x : numpy.array of inputs to NN. Material parameters in their standard normalized form(par_norm). Axis = 0 should be equal to axis = 0 of y.
    y : numpy.array of y-axis values. Each row is new result.  Axis = 0 should be equal to axis = 0 of x. axis = 1 must have 4 points
    training folder : folder where the training results will be stored after completion of training.
    training_name : name of hdf file in which the weights and biases will be stored

    Saves
    -------
    reg : regression model
    reg_name : path where the regression model is stored
    x_test : the x values used for testing
    y_test : the y values used for testing
    x_train : the x_values used for training and validation
    y_train : the y values used for training and validation
    '''

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.10, shuffle=False)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # Network Parameters
    max_filter = 64
    strides = [2,2]
    kernel = [2,2]
    map_size = 1

    nParams = X_train.shape[1]

    z_in = tf.keras.layers.Input(shape=(nParams,))
    z1 = tf.keras.layers.Dense(max_filter)(z_in)
    z1 = tf.keras.activations.swish(z1)
    z1 = tf.keras.layers.Dense(max_filter*map_size)(z1) #64 * 1
    z1 = tf.keras.activations.swish(z1)
    z1 = tf.keras.layers.Reshape((map_size,max_filter))(z1) # 1 by 64
    z2 = tf.keras.layers.Conv1DTranspose( max_filter//2, kernel[1], strides=strides[1], padding='SAME')(z1) # 2 by 32
    z2 = tf.keras.activations.swish(z2)
    z3 = tf.keras.layers.Conv1DTranspose(1, kernel[0], strides=strides[0],padding='SAME')(z2) # 4 by 1
    decoded_Y = tf.keras.activations.swish(z3)
    decoded_Y = tf.keras.layers.Reshape((Y_train.shape[1],))(decoded_Y)
    
    # in case the learning rate is not constant, define the scheduler here
    def scheduler(epoch, lr):
        if epoch <120:  # number of epoches, where the first learning rate is valid
            lr = lrate
            return lr
        else :
            lr = lrate # decrease learning rate with every step
            return lr
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)   # create the tensorflow object for the scheduler 
      
    log_folder = training_folder /  training_name # define folder where data for tensorboard and checkpoint are stored
    print(log_folder)   # print folder name so it can be copied for tensorboard
    tf_callbacks = [TensorBoard(log_dir=log_folder,     # allow tensorflow to store information for tensorboard
                        histogram_freq=1,
                        update_freq='epoch',
                         profile_batch=(2,10))]
    # to activate tensorboard to observe the training losses, activate the right environment in conda and use the command 'tensorboard --logdir log_folder'

    # currently, checkpoints are saved after each iteration, I might uncomment this. To load the checkpoint, use 'reg.load_weights(checkpoint_path)'
    # checkpoint_path = log_folder / ("cp.ckpt")  # define paths where the checkpoints are stored
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,  verbose=1)  # define how checkpoints are made

    reg = Model(z_in,decoded_Y) # create a model with the layers defined before
    reg.summary()   # print information on model
    
    # the neural network can stop before the full number of epoches are run. Here, the criterion for this is the validation loss. 
    # If it has not varied by the min_delta, it will run another 'patience' epochs before terminating the training. Then the best 
    # weights will be used based on the mode.
    es = EarlyStopping(monitor='val_loss', min_delta=5e-10, mode='min', verbose=1, patience=200, restore_best_weights=True) 
    
    reg.compile(loss='mse',optimizer='adam', metrics=['mae', 'msle'])   # mean square error is used for training with the optimizer 'adam'

    # run the training of the neural network.
    reg.fit(X_train,Y_train,shuffle=False, batch_size=batchsize, epochs = 10000,
            validation_split=0.4,  callbacks=[lr_callback, tf_callbacks], verbose = 0)
        
    reg_name = Path(training_folder)/("%s_trained_model.h5" % training_name)    # create path to store the neural network
    # save the neural network
    reg.save(reg_name)  
    reg.save(Path(training_folder)/("model" ))
    return reg, reg_name, X_test, Y_test, X_train, Y_train


def log_trans(paras, n_var):
    '''
    log_trans : take the logarithm of a certain subset of data that is varied over several orders of magnitude  
                !!! number of parameters that scales linearly must be defined here !!!
    
    Parameters
    ----------
    paras : numpy.array of data of which a subset is supposed to be transformed. one dimension must be of the size n_var 
    n_var : number of variable parameters

    Returns
    -------
    mod : data with the natural logarithm of some data
    ''' 
    if numpy.shape(paras)[0] == n_var:  # check which dimension contains the n_var parameters
        mod = numpy.concatenate((paras[:3],numpy.log(paras[3:])),axis=0)    # take logarithm of some parameters along that axis
    elif numpy.shape(paras)[1] == n_var:    # check which dimension contains the n_var parameters
        mod = numpy.concatenate((paras[:,:3],numpy.log(paras[:,3:])),axis=1)     # take logarithm of some parameters along that axis
    return mod

def exp_trans(paras, n_var):
    '''
    exp_trans : take the exponential of a certain subset of data that is varied over several orders of magnitude  
                !!! number of parameters that scales linearly must be defined here !!!
    
    Parameters
    ----------
    paras : numpy.array of data of which a subset is supposed to be transformed. one dimension must be of the size n_var 
    n_var : number of variable parameters

    Returns
    -------
    mod : data with the exponential of some data
    ''' 
    if numpy.shape(paras)[0] == n_var:  # check which dimension contains the n_var parameters
        mod = numpy.concatenate((paras[:3],numpy.exp(paras[3:])),axis=0)    # take exponential of some parameters along that axis
    elif numpy.shape(paras)[1] == n_var:  # check which dimension contains the n_var parameters
        mod = numpy.concatenate((paras[:,:3],numpy.exp(paras[:,3:])),axis=1)    # take exponential of some parameters along that axis
    return mod

def main(sim_name, timestamp, dataset, lr, batch_size):
    
    dir, name = create_dir(sim_name, timestamp) # create directory for this training run
    print('DIRECTORY CREATED AT:', dir)
    
    backup_path = dir/("ScriptsUsed") # create folder path to copy the python script version of this run to 
    os.mkdir(backup_path)   # create folder
    shutil.copyfile("run.py", backup_path/("run.py"))   # copy main run file to backup folder
    shutil.copyfile("NN_training.py", backup_path/"NN_training.py") # copy NN training python script to backup folder


    train_data = H5()   # create a h5 object for the training data
    train_name = dir/("%s_train_test.h5" % name)   # name for the file where all your training related data is getting stored
    train_data.filename = train_name.as_posix() # write filename into h5 objject
    train_data.root.dataset = str(dataset)  # save original folder of the training data
    train_data.save()   # save h5 file for training data
   
    y1_norm, y2_norm, y1_max, y1_min, y2_max, y2_min, par_mat, ub, lb, n_var = load_dataset(dataset)    # load training dataset
    # save maxima and minima of training to be able to rescale data later
    train_data.root.y1_max = y1_max
    train_data.root.y1_min = y1_min
    train_data.root.y2_max = y2_max
    train_data.root.y2_min = y2_min
    # save upper and lower boundaries of the variable parameters and the number of parameters
    train_data.root.ub = ub
    train_data.root.lb = lb
    train_data.root.n_var = n_var
    print('DATA LOADED FROM:', dataset)
    
    
    par_norm, scaler = par_fit_transform(par_mat, n_var)       # StandardScaler transform par_mat to fit the Neural Network
    train_data.root.par_norm = par_norm     # store transformed parameter matrix in h5 object
    scaler_name = dir/(name + "scaler.joblib")  # create filname for the scaler used
    joblib.dump(scaler, scaler_name)    # save scaler at the path 'scaler_name'

    ### evaluate first data set
    # select some random points and plot to see if you have loaded the correct datasets.
    dir1 = dir/('y1')
    os.mkdir(dir1)  # create directory for the first neural network
    idx = numpy.random.randint(0, y1_norm.shape[0],9)   # get random indices for test ploting of the training data
    plot_sim(y1_norm[idx,:], dir1, 'y1_norm.png')   # plot test subset of the training data

    # train neural network
    name1 = name + '_y1'    # define name for training folder of this neural network
    reg1, reg_name1, x1_test, y1_test, x1_train, y1_train = network512(par_norm, y1_norm, dir1, name1, lr, batch_size) # run neural network
    # save test split of the training data which was not used for the training of the neural network
    train_data.root.X1_test = x1_test
    train_data.root.Y1_test = y1_test
    # save train split of the training data which was used for the training of the neural network
    train_data.root.X1_train = x1_train
    train_data.root.Y1_train = y1_train
    
    train_data.root.reg_name1 = str(reg_name1)# save the path for the first neural network
    train_data.root.train_name = str(train_name)    # save the path for the training data
    train_data.root.scaler_name = str(scaler_name)  # save the path for the scaler
    train_data.save()   # save h5 object
    # reg_name1 = 'bla'
    
    ### evaluate second data set
    # select some random points and plot to see if you have loaded the correct datasets.
    dir2 = dir/('y2')
    os.mkdir(dir2)
    num = 9
    idx = numpy.random.randint(0, y2_norm.shape[0],num)
    plot_sim(y2_norm[idx,:], dir2, 'y2_norm.png')
    # train neural network
    loadweights = 'no'
    name2 = name + '_y2'
    reg2, reg_name2, x2_test, y2_test, x2_train, y2_train = network4(par_norm, y2_norm, dir2, name2, lr, batch_size)
    train_data.root.X2_test = x2_test
    train_data.root.Y2_test = y2_test
    train_data.root.X2_train = x2_train
    train_data.root.Y2_train = y2_train
    train_data.root.reg_name2 = str(reg_name2)
    train_data.save()

    return dir, reg_name1, reg_name2, train_name, scaler_name


