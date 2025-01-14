#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import random 
import argparse
import logging 
import os 
import csv
import numpy as np 
import torch 
import torch.nn as nn 
from torch import optim 
from tqdm import tqdm 
import wandb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
import importlib
from utils.eval_metrics import eval_metrics
from utils.save_metrics import save_metrics
from utils.import_helper import import_config
from utils.TrainUtils import create_directory
from utils.dataset_main import BasicDataset as BasicDataset_CSV
from utils.dataset_teacher import BasicDataset as BasicDataset_CSV_Teacher
from torch.utils.data import DataLoader
from utils_SemiSup.pseudo_labeler_DIST import pseudo_labeler_DIST_Stage1, pseudo_labeler_DIST_Stage2

class printer(nn.Module):
        def __init__(self, global_dict=globals()):
            super(printer, self).__init__()
            self.global_dict = global_dict
            self.except_list = []
        def debug(self,expression):
            frame = sys._getframe(1)

            print(expression, '=', repr(eval(expression, frame.f_globals, frame.f_locals)))

        def namestr(self,obj, namespace):
            return [name for name in namespace if namespace[name] is obj]     
        
        def forward(self,x):
            for i in x:
                if i not in self.except_list:
                    name = self.namestr(i, globals())
                    if len(name)>1:
                        self.except_list.append(i)
                        for j in range(len(name)):
                            self.debug(name[j])
                    else:  
                        self.debug(name[0])

def train_net(net_student,
              net_teacher,
              epochs=40,
              batch_size=16,
              lr=0.001,
              device='cuda',
              save_cp=True,
              Categories = ''
              ):
    """
    Args:
        net_student: The student network to be trained.
        net_teacher: The teacher network for pseudo-label generation.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        device: Device to use for training ('cuda' or 'cpu').
        save_cp: Whether to save model checkpoints.
        Categories: Category information for training.
    """
    
    TESTS = []

    # Load datasets
    train_dataset = BasicDataset_CSV_Teacher(train_IDs_CSV, semi_IDs_CSV) # labeled dataset
    test_dataset = BasicDataset_CSV(test_IDs_CSV, doTransform = False)
    semi_dataset = BasicDataset_CSV(semi_IDs_CSV, doTransform = False) # unlabeled dataset

    n_train = len(train_dataset)  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=False)
    n_test = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
    n_semi = len(semi_dataset)
    semi_loader = DataLoader(semi_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=False) 
   
    global_step = 0
    total_iters = len(train_loader) * epochs*epoch_multiplier
    inference_step = np.floor(np.floor(n_train/batch_size)/(test_per_epoch))*epoch_multiplier
    print(f'Inference Step:{inference_step}')

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Test size:       {n_test}
        Semi size:       {n_semi}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    # Optimizer setup
    optimizer = optim.SGD([{'params': net_student.feature_extractor.parameters(), 'lr': lr},
                     {'params': [param for name, param in net_student.named_parameters()
                                 if 'feature_extractor' not in name],
                      'lr': lr}], lr=lr, momentum=0.9, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    test_counter = 1

    for epoch in range(epochs*epoch_multiplier):
        net_student.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs*epoch_multiplier}', unit='frames') as pbar:
            for batch in train_loader:

                vids = batch['video']
                true_labels = batch['label']
                names = batch['name']

                # Forward pass
                labels_pred = net_student(vids).view(-1,4)
                loss_main = criterion(labels_pred, true_labels.view(-1))
                loss_wandb = loss_main
                loss = loss_main
                epoch_loss += loss.item()
                
                # Backpropagation
                optimizer.zero_grad()            
                (loss_main).backward()
                optimizer.step()

                # Learning rate adjustment
                lr1 = lr * (1 - global_step / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr1

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(vids.shape[0])               
                global_step += 1

                # Evaluation metrics
                if (global_step) % (inference_step) == 0: 
                    for tag, value in net_student.named_parameters():
                        tag = tag.replace('.', '/')
    
                    accuracy, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1, confusion_matrix, inference_time = eval_metrics(
                        net_student, test_loader, device, test_counter, save_test, save=False)
                    
                    print(f'Accuracy:{accuracy}')
                    print(f'weighted_recall:{weighted_recall}')
                    print(f'weighted_precision:{weighted_precision}')
                    print(f'weighted_f1:{weighted_f1}')
                    print (f'confusion_matrix:{confusion_matrix}')

                    TESTS.append([accuracy, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1, confusion_matrix, inference_time, epoch_loss])
                    test_counter = test_counter+1                       
                      
                    wandb.log({'Train_Loss': loss_wandb,
                               'Test_Accuracy': accuracy,
                               'Test_f1score': weighted_f1})

        # Save checkpoint
        if save_cp and (epoch + 1) == epochs*epoch_multiplier:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            
            torch.save(net_student.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    # Final evaluation           
    accuracy, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1, confusion_matrix, inference_time = eval_metrics(
        net_student, test_loader, device, test_counter, save_test, save=False)
    save_metrics(TESTS, csv_name)


if __name__ == '__main__':
    args = parser.parse_args()
    config_file = args.config
    my_conf = importlib.import_module(config_file)
    Categories, Learning_Rates_init, epochs, batch_size, size,\
             Dataset_Path_Train, Dataset_Path_SemiTrain, Dataset_Path_Test,\
                  mask_folder, Results_path, Visualization_path, error_path,\
                    CSV_path,  TrainIDs_path, project_name, load, load_path, net_name,\
                      test_per_epoch, Checkpoint_path, Net1,\
                         hard_label_thr, ensemble_batch_size, SemiSup_initial_epoch,\
                            image_transforms, affine, affine_transforms, LW,\
                                 EMA_decay, Alpha, strategy, GCC, supervised_share\
                     = import_config.execute(my_conf)
    
    # Device setup
    print("inside main")
    print(f'Cuda Availability: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')
    print(f'Cuda Device Number: {torch.cuda.current_device()}')
    print(f'Cuda Device Name: {torch.cuda.get_device_name(0)}')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    printer1 = printer()       
    
    print('CONFIGS:________________________________________________________')
    
    printer1([Categories, Learning_Rates_init, epochs, batch_size, size,\
              Dataset_Path_Train, Dataset_Path_SemiTrain, Dataset_Path_Test,\
                mask_folder, Results_path, Visualization_path, error_path,\
                    CSV_path,  TrainIDs_path, project_name, load, load_path, net_name,\
                        test_per_epoch, Checkpoint_path, Net1,\
                            hard_label_thr, ensemble_batch_size, SemiSup_initial_epoch,\
                                image_transforms, affine, affine_transforms, LW,\
                                    EMA_decay, Alpha, strategy, GCC, supervised_share])  
    
    
    try:
        # DIST Stage 1
        for c in range(len(Categories)):      
            for LR in range(len(Learning_Rates_init)):
                print(f'Initializing the learning rate: {Learning_Rates_init[LR]}')
                epoch_multiplier = Categories[c][-1]

                # Initialize WandB for tracking
                run = wandb.init(project=project_name+'_'+net_name+'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size), entity="s-nasirihg",
                name=strategy+'_stage1_'+Categories[c][0][:-4])
                wandb.config = {
                "learning_rate": Learning_Rates_init[LR],
                "epochs": epochs,
                "batch_size": batch_size,
                "net_name": net_name,
                "Dataset": "Fold"+str(c),
                }

                # Define file paths
                train_IDs_CSV  =  TrainIDs_path + Categories[c][0]
                semi_IDs_CSV =  TrainIDs_path + Categories[c][1]
                test_IDs_CSV =  TrainIDs_path + Categories[c][2]

                save_test = Results_path + Visualization_path + project_name + '_' + strategy +'_stage1_'+net_name +'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c][0][:-4])+'_'+'/'
                dir_checkpoint = Results_path + Checkpoint_path + project_name + '_'+ strategy + '_stage1_'+net_name +'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c][0][:-4])+'/'
                csv_name = Results_path + CSV_path + project_name + '_' + strategy +'_'+net_name +'_stage1_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c][0][:-4])+'_'+'.csv'
                error_name = Results_path + error_path + project_name + '_'+ strategy + '_stage1_'+net_name +'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c][0][:-4])+'.err'
            
                create_directory(Results_path + Visualization_path)
                create_directory(Results_path + Checkpoint_path)
                create_directory(Results_path + CSV_path)

                # Initialize networks
                net_student = Net1(n_classes=4)
                net_teacher1 = Net1(n_classes=4)
                net_teacher2 = Net1(n_classes=4)
                net_teacher3 = Net1(n_classes=4) 

                net_student.to(device=device)                
                net_teacher1.to(device=device)
                net_teacher2.to(device=device)
                net_teacher3.to(device=device)

                # Load teacher network (3 different checkpoints)
                loading_epochs = [epochs//3, 2*epochs//3, epochs]
                if load:                    
                    load_path_1 = Results_path + Checkpoint_path + project_name + '_'+ 'Supervise' + '_'+net_name +'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c][0][:-4])+'/'+'CP_epoch'+str(loading_epochs[0])+'.pth'
                    net_teacher1.load_state_dict(torch.load(load_path_1 , map_location=device))    
                    net_teacher1.eval()

                    load_path_2 = Results_path + Checkpoint_path + project_name + '_'+ 'Supervise' + '_'+net_name +'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c][0][:-4])+'/'+'CP_epoch'+str(loading_epochs[1])+'.pth'
                    net_teacher2.load_state_dict(torch.load(load_path_2 , map_location=device))    
                    net_teacher2.eval()

                    load_path_3 = Results_path + Checkpoint_path + project_name + '_'+ 'Supervise' + '_'+net_name +'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c][0][:-4])+'/'+'CP_epoch'+str(loading_epochs[2])+'.pth'
                    net_teacher3.load_state_dict(torch.load(load_path_3 , map_location=device))    
                    net_teacher3.eval()
                    
                    # Generate pseudo labels for Stage 1
                    csv_name_new = 'DIST_stage1_'+ net_name +'_'+Categories[c][1]  
                    pseudo_labeler_DIST_Stage1(net_teacher1, net_teacher2, net_teacher3, semi_IDs_CSV, TrainIDs_path, csv_name_new)
                    semi_IDs_CSV = TrainIDs_path + csv_name_new

                # Train student network                    
                train_net(net_student=net_student,
                        net_teacher=net_teacher1,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=Learning_Rates_init[LR],
                        device=device,
                        Categories = Categories[c])
                                
        run.finish()  

        # DIST Stage 2
        for c in range(len(Categories)):      
            for LR in range(len(Learning_Rates_init)):
                print(f'Initializing the learning rate: {Learning_Rates_init[LR]}')
                epoch_multiplier = Categories[c][-1]

                # Initialize WandB for tracking
                run = wandb.init(project=project_name+'_'+net_name+'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size), entity="s-nasirihg",
                name=strategy+'_stage2_'+Categories[c][0][:-4])
                wandb.config = {
                "learning_rate": Learning_Rates_init[LR],
                "epochs": epochs,
                "batch_size": batch_size,
                "net_name": net_name,
                "Dataset": "Fold"+str(c),
                }

                # Define file paths
                train_IDs_CSV  =  TrainIDs_path + Categories[c][0]
                semi_IDs_CSV =  TrainIDs_path + Categories[c][1]
                test_IDs_CSV =  TrainIDs_path + Categories[c][2]

                save_test = Results_path + Visualization_path + project_name + '_' + strategy +'_stage2_'+net_name +'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c][0][:-4])+'_'+'/'
                dir_checkpoint = Results_path + Checkpoint_path + project_name + '_'+ strategy + '_stage2_'+net_name +'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c][0][:-4])+'/'
                csv_name = Results_path + CSV_path + project_name + '_' + strategy +'_'+net_name +'_stage2_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c][0][:-4])+'_'+'.csv'
                error_name = Results_path + error_path + project_name + '_'+ strategy + '_stage2_'+net_name +'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c][0][:-4])+'.err'
                            
                create_directory(Results_path + Visualization_path)
                create_directory(Results_path + Checkpoint_path)
                create_directory(Results_path + CSV_path)

                # Initialize networks
                net_student = Net1(n_classes=4)
                net_teacher1 = Net1(n_classes=4)
                
                net_student.to(device=device)                
                net_teacher1.to(device=device)
             
                # Load teacher network from Stage 1
                loading_epochs = [epochs//3, 2*epochs//3, epochs]
                if load:
                    load_path_1 = Results_path + Checkpoint_path + project_name + '_'+ strategy + '_stage1_'+net_name +'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c][0][:-4])+'/'+'CP_epoch'+str(loading_epochs[2])+'.pth'                    
                    net_teacher1.load_state_dict(torch.load(load_path_1 , map_location=device))
                    net_teacher1.eval()

                    # Generate pseudo labels for Stage 2
                    csv_name_new = 'DIST_stage2_'+ net_name +'_'+Categories[c][1] 
                    pseudo_labeler_DIST_Stage2(net_teacher1, semi_IDs_CSV, TrainIDs_path, csv_name_new)
                    semi_IDs_CSV = TrainIDs_path + csv_name_new

                # Train student network
                train_net(net_student=net_student,
                        net_teacher=net_teacher1,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=Learning_Rates_init[LR],
                        device=device,
                        Categories = Categories[c])
                
        run.finish()             

    except KeyboardInterrupt:
            torch.save(net_student.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)