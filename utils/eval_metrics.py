# -*- coding: utf-8 -*-

import torch
from torchmetrics.functional import accuracy, f1_score, recall, precision
from torchmetrics.functional.classification import multiclass_confusion_matrix
import os
import timeit

def eval_metrics(net, loader, device, test_counter, save_dir, rep_head=False, save=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    Inference_time = []
    label_pred_list = []
    true_labels_list = []

    try:
       os.mkdir(save_dir)
    except OSError:
       pass        
    for batch in loader:
        try:
            vids, true_labels, name = batch['video'], batch['label'], batch['name']
        except KeyError:
            vids, true_labels, name = batch['video_teacher'], batch['label'], batch['name']
        
        start = timeit.default_timer()            
        with torch.no_grad():

            if rep_head==True:
                labels_pred, rep = net(vids)
            else:
                labels_pred = net(vids) 

            labels_pred = torch.mean(labels_pred, dim=1) # one label for each video
        stop = timeit.default_timer()
        Inference_time.append(stop - start)
        true_labels_list.append(true_labels[:, 0]) 
        label_pred_list.append(labels_pred)
        

    true_labels = (torch.stack([true_labels_list[i] for i in range(len(true_labels_list))]).view(-1))
    label_pred = (torch.stack([label_pred_list[i] for i in range(len(label_pred_list))]).view(-1, 4))

    Accuracy = float(accuracy(label_pred, true_labels, task="multiclass", num_classes=4))
    macro_Precision = float(precision(label_pred, true_labels, average='macro', task="multiclass", num_classes=4))
    macro_Recall = float(recall(label_pred, true_labels, average='macro', task="multiclass", num_classes=4))
    macro_f1Score = float(f1_score(label_pred, true_labels, average='macro', task="multiclass", num_classes=4))

    weighted_precision = float(precision(label_pred, true_labels, average='weighted', task="multiclass", num_classes=4))
    weighted_recall = float(recall(label_pred, true_labels, average='weighted', task="multiclass", num_classes=4))
    weighted_f1 = float(f1_score(label_pred, true_labels, average='weighted', task="multiclass", num_classes=4))

    confusion_matrix = multiclass_confusion_matrix(label_pred, true_labels, num_classes=4)

    net.train()
    return Accuracy, macro_Precision, macro_Recall, macro_f1Score, weighted_precision, weighted_recall, weighted_f1, confusion_matrix, sum(Inference_time)/len(Inference_time)