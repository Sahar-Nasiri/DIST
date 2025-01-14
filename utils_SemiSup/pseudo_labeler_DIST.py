
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import pandas as pd
from utils.dataset_semi import BasicDataset as BasicDataset_CSV
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import csv

def pseudo_labeler_DIST_Stage1(net_teacher1, net_teacher2, net_teacher3, semi_IDs_CSV, train_IDs_CSV, csv_name_semiID, eps = 1e-6):     
    """
    Generate three sets of pseudo labels using three different checkpoints of the teacher model for semi-supervised learning (Stage 1).

    Args:
        net_teacher1, net_teacher2, net_teacher3: three checkpoints of teacher model for generating pseudo labels.
        semi_IDs_CSV: Path to the CSV file containing unlabeled data IDs.
        train_IDs_CSV: Path to the CSV file for saving generated pseudo labels.
        csv_name_semiID: Filename for the output pseudo-labeled CSV file.
        eps: Small value to prevent division by zero.
    """
    # stage 1
    # 1) loading the dataset
    # Dual Invariance: 1.temporal invariance --> randomly sampled sequence
    SemiSup_dataset = BasicDataset_CSV(semi_IDs_CSV, doTransform = False) 
    SemiSup_loader = DataLoader(SemiSup_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=True)
    n_semi = len(SemiSup_loader)

    vid_list = []
    label_list = []
    pseudo_labels = []
    pseudo_labels_aug = []
    IoUs= []
    names= []
   
    for batch in SemiSup_loader:           
            vids, vids_aug, _, name = batch['video'], batch['video_aug'], batch['label'], batch['name']    
            with torch.no_grad():
                  label_prob1 = net_teacher1(vids)
                  label_prob2 = net_teacher2(vids)
                  label_prob3 = net_teacher3(vids)
                  label_prob_aug = net_teacher3(vids_aug)

            label_prob1 = F.softmax(label_prob1, dim=-1) 
            label_prob2 = F.softmax(label_prob2, dim=-1)
            label_pred3 = torch.argmax(F.softmax(label_prob3, dim=-1), dim=-1)
            
            pseudo_labels.append(torch.argmax(torch.mean(F.softmax(label_prob3, dim=-1), dim = 1), dim=-1)) # bs*frames*classes --> bs*1*classes
            pseudo_labels_aug.append(torch.argmax(torch.mean(F.softmax(label_prob_aug, dim=-1), dim = 1), dim=-1))
            
            # 2) measure pseudo-label consistency
            label_pred3 = F.one_hot(label_pred3,  num_classes=4)            
            intersection13 = torch.sum(label_prob1 * label_pred3)
            denominator13 = torch.sum(label_prob1 + label_pred3) - intersection13
            intersection23 = torch.sum(label_prob2 * label_pred3)
            denominator23 = torch.sum(label_prob2 + label_pred3) - intersection23
            IoU = ((intersection13) / (denominator13 + eps) + 2*(intersection23) / (denominator23 + eps))/3

            IoUs.append(IoU)
            names.append(name)

    # 3) retain the top 50% of pseudo-labels       
    l = len(IoUs)
    l1 = l//2
    IoUs = torch.tensor(IoUs)   
    stable_indices = torch.topk(IoUs, l1).indices

    si = 0
    for batch in SemiSup_loader:    
        if si in stable_indices:
            # Dual Invariance: 2.transformation invariance --> strongly augmented version 
            if pseudo_labels[si] == pseudo_labels_aug[si]:

                vid_list.append(names[si][0])
                label_list.append(pseudo_labels[si][0].cpu().numpy())
        si += 1
    
         
    # 4) csv file creator
    dict = {"vids": vid_list , "labels": label_list}
    df = pd.DataFrame(dict)
  
    df.to_csv(train_IDs_CSV + csv_name_semiID)
    return SemiSup_loader, n_semi

# stage 2
def pseudo_labeler_DIST_Stage2(net_teacher, semi_IDs_CSV, train_IDs_CSV, csv_name_semiID, threshold=0):
    # 1) loading the dataset
    # Dual Invariance: 1.temporal invariance --> randomly sampled sequence
    SemiSup_dataset = BasicDataset_CSV(semi_IDs_CSV, doTransform = False) 
    SemiSup_loader = DataLoader(SemiSup_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=True)

    vid_list = []
    label_list = []
    
    for batch in SemiSup_loader:
            vids, vids_aug, _, name = batch['video'], batch['video_aug'], batch['label'], batch['name']            
            with torch.no_grad():
                label_prob = net_teacher(vids)
                label_prob_aug2 = net_teacher(vids_aug)
            
            label_prob = torch.softmax(label_prob, dim=-1)
            prob_avg = torch.mean(label_prob, dim=1)
            label_pred = torch.argmax(prob_avg, dim=-1)
            
            label_prob_aug2 = F.softmax(label_prob_aug2, dim=-1)
            prob_avg_aug2 = torch.mean(label_prob_aug2, dim=1)
            label_pred_aug2 = torch.argmax(prob_avg_aug2, dim=-1)
            # Dual Invariance: 2.transformation invariance --> strongly augmented version 
            if label_pred == label_pred_aug2:
                vid_list.append(name[0])
                label_list.append(label_pred[0].cpu().numpy())

    # 4) csv file creator
    dict = {"vids": vid_list , "labels": label_list}
    df = pd.DataFrame(dict)
    df.to_csv(train_IDs_CSV + csv_name_semiID)

    return 

