import NN_training
from pathlib import Path
import datetime


dataset = Path("Dataset/20241117-151858-8_param_biOD_norm.h5")  # path for the transformed training data
lr = 0.0001 # only used for nn2!
batch_size = 1024
# batch_size_arr = numpy.round(numpy.logspace(2,4,3))ex
# for batch_size in batch_size_arr:
NN_name = '_binary_OD'  # name for NN training run

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")   # get initial time for unique filenames
dir_path, reg1_path, reg2_path, train_path, scaler_path = NN_training.main(NN_name, timestamp, dataset, lr, int(batch_size))    # run neural network training
# print relevant paths that can be later inserted in the next section
print('reg_path', reg1_path, reg2_path)
print('train_path', train_path)
print('scaler_path', scaler_path)