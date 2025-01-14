
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from .Transforms import *
import pandas as pd
import cv2
from torchvision.transforms import *
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import random


class BasicDataset(Dataset):
    def __init__(self, train_IDs_CSV, size = (256, 256),
                 device = 'cuda' if torch.cuda.is_available() else 'cpu',
                 doTransform = True, strategy='default'):
        
        self.train_IDs_CSV = train_IDs_CSV
        self.device = device
        self.size = size
        self.doTransform = doTransform
        self.strategy = strategy        

        data = pd.read_csv(self.train_IDs_CSV, usecols = ['vids','labels'])

        self.ids_vids = data['vids'].tolist()
        self.ids_labels = data['labels'].tolist()
        if len(self.ids_labels) == 0:
            self.ids_labels = 100 * np.ones_like(len(self.ids_vids))

        logging.info(f'Creating dataset with {len(self.ids_vids)} examples')
        
        self.transforms = v2.Compose([v2.Resize(self.size),
                                      v2.RandomRotation(degrees=15),
                                      v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.5, hue=0),
                                      v2.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 2.0)),
                                      ]) 
       
        self.transforms_necessary = v2.Compose([v2.Resize(self.size),
                                                ])


    def __len__(self):
        return len(self.ids_vids)
    

    def load_video_uniform(self, idx, num_frames=10, resize=(256, 256)):
        """Load video frames uniformly across the video."""
        cap = cv2.VideoCapture(self.ids_vids[idx])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        num_frames = min(num_frames, total_frames)  

        frame_step = total_frames // num_frames

        frames = []
        labels = []
        try:
            for i in range(num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i*frame_step)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
                labels.append(torch.Tensor(np.array([self.ids_labels[idx]])))
        finally:
            cap.release()
         
        return frames, labels
    
    def load_video_random(self, idx, num_frames=10, resize=(256, 256)):
        """Load random frames from the video."""
        cap = cv2.VideoCapture(self.ids_vids[idx])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        num_frames = min(num_frames, total_frames)  

        random_frames = random.sample(range(total_frames), num_frames)

        frames = []
        labels = []
        try:
            for i in range(num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, random_frames[i])
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
                labels.append(torch.Tensor(np.array([self.ids_labels[idx]])))           
        finally:
            cap.release()
        return frames, labels
        

    def __getitem__(self, idx):
        frames1, labels = self.load_video_uniform(idx)
        frames_no_aug = self.transforms_necessary(torch.from_numpy(np.array(frames1)/256).to(self.device).permute(0, 3, 1, 2))
        frames_no_aug = torch.stack([frames_no_aug[i] for i in range(len(frames_no_aug))])

        frames1, labels = self.load_video_random(idx)
        frames_augmented = self.transforms(torch.from_numpy(np.array(frames1)/256).to(self.device).permute(0, 3, 1, 2))
        frames_augmented = torch.stack([frames_augmented[i] for i in range(len(frames_augmented))])

        
        labels = torch.stack([labels[i] for i in range(len(labels))])
        labels = labels.long().squeeze(-1)
        

        return {
            'video': frames_no_aug.type(torch.cuda.FloatTensor),
            'video_aug': frames_augmented.type(torch.cuda.FloatTensor),
            'label': labels.type(torch.cuda.LongTensor),
            'name': str(self.ids_vids[idx])
        }


if __name__ == '__main__':
    train_IDs_CSV = "/TrainIDs_Cataract_Videos/Cataract_split_2_test_multi_0.csv"
    train_dataset = BasicDataset(train_IDs_CSV, doTransform= False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=False)
    n_train = len(train_dataset)
    train_loader_iterator = iter(train_loader)
    for i in range(10):
        new_batch = next(train_loader_iterator)
        nb = new_batch['video']  
        nl = new_batch['label']
        nn = new_batch['name']
        print(f'name: {nn}')
        print(f'label: {nl}')
      
