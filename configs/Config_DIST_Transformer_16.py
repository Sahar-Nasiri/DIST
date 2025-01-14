#############################################################
# Importing from a sibling directory:
import sys
sys.path.append("..")
#############################################################

from utils.Transforms import *
from nets import VGG_Transformer as Net

Net1 = Net

Categories = [
['Cataract_split_16_train_multi_0.csv', 'Cataract_split_16_semi_multi_0.csv', 'Cataract_split_16_test_multi_0.csv', 1],
['Cataract_split_16_train_multi_1.csv', 'Cataract_split_16_semi_multi_1.csv', 'Cataract_split_16_test_multi_1.csv', 1],
['Cataract_split_16_train_multi_2.csv', 'Cataract_split_16_semi_multi_2.csv', 'Cataract_split_16_test_multi_2.csv', 1],
['Cataract_split_16_train_multi_3.csv', 'Cataract_split_16_semi_multi_3.csv', 'Cataract_split_16_test_multi_3.csv', 1]
]

Learning_Rates_init = [0.005]
epochs = 40
batch_size = 16
size = (256, 256)

Dataset_Path_Train = '/home/itec/sahar/Domain_Adaptation/DA_main/multiclass_cataract_4class/'
Dataset_Path_SemiTrain = ''
Dataset_Path_Test = ''
mask_folder = ''
Results_path = '/home/itec/sahar/Domain_Adaptation/DA_main/Results/'
Visualization_path = 'visualization/'
Checkpoint_path = 'checkpoints/'
CSV_path = 'CSVs_RETOUCH/'
TrainIDs_path = 'TrainIDs_Cataract_Videos/'
error_path = 'code_errors/'
project_name = "DA_SMISUP_CATARACT_MIC24"

hard_label_thr = ''

# Warning: if the model weights are loaded, the learning rate should also change based on the number of epochs
load = True
load_path = ''
load_epoch = ''

net_name = 'VGG_Transformer'
strategy = "DIST"
test_per_epoch = 2

ensemble_batch_size = 16
SemiSup_initial_epoch = 0
supervised_share = 1

image_transforms = ''

affine = ''
affine_transforms = ''

# Unsupervised loss-weightening function parameters:  
LW = 1
GCC = 2
Alpha = 1

# Unsupervised average-mask weightening function parameters:
EMA_decay = 0.99