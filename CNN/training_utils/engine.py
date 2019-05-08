import copy
import re

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


import numpy as np

from statistics import mean

import shutil
import os

import sklearn
from sklearn.metrics import roc_curve


from iotools.data_handling import WCH5Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class Engine:
    """The training engine 
    
    Performs training and evaluation
    """

    def __init__(self, model,config):
        self.model = model

        if config.gpu:
            print("requesting gpu ")
            print("gpu list: ")
            print(config.gpu_list)
            self.devids = ["cuda:{0}".format(x) for x in config.gpu_list]

            print("main gpu: "+self.devids[0])
            if torch.cuda.is_available():
                self.device = torch.device(self.devids[0])
                if len(self.devids) > 1:
                    print("using DataParallel on these devices: {}".format(self.devids))
                    self.model = nn.DataParallel(self.model, device_ids=config.gpu_list, dim=0)

                print("cuda is available")
            else:
                self.device=torch.device("cpu")
                print("cuda is not available")
        else:
            print("will not use gpu")
            self.device=torch.device("cpu")

        print(self.device)

        self.model.to(self.device)

        self.opt = opt.Adam(self.model.parameters(),eps=1e-3)
        self.crit = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        #placeholders for data and labels
        self.data=None
        self.labels=None
        self.iteration=None

        self.dset=WCH5Dataset(config.path, config.val_split, config.test_split)

        self.train_iter=DataLoader(self.dset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   sampler=SubsetRandomSampler(self.dset.train_indices))
        
        self.val_iter=DataLoader(self.dset,
                                 batch_size=config.batch_size_val,
                                 shuffle=True,
                                 sampler=SubsetRandomSampler(self.dset.val_indices))
        
        self.test_iter=DataLoader(self.dset,
                                  batch_size=config.batch_size_val,
                                  shuffle=True,
                                  sampler=SubsetRandomSampler(self.dset.test_indices))

        

        self.dirpath=config.save_path
        
        self.data_description=config.data_description


        
        try:
            os.stat(self.dirpath)
        except:
            print("making a directory for model data: {}".format(self.dirpath))
            os.mkdir(self.dirpath)

        #add the path for the data type to the dirpath
        self.start_time_str = time.strftime("%Y%m%d_%H%M%S")
        self.dirpath=self.dirpath+'/'+self.data_description + "/" + self.start_time_str

        try:
            os.stat(self.dirpath)
        except:
            print("making a directory for model data for data prepared as: {}".format(self.data_description))
            os.makedirs(self.dirpath,exist_ok=True)

        self.config=config



    def forward(self,train=True):
        """
        Args: self should have attributes, model, criterion, softmax, data, label
        Returns: a dictionary of predicted labels, softmax, loss, and accuracy
        """
        with torch.set_grad_enabled(train):
            # Prediction
            #print("this is the data size before permuting: {}".format(data.size()))
            data = data.permute(0,3,1,2)
            #print("this is the data size after permuting: {}".format(data.size()))
            prediction = self.model(data)
            # Training
            loss,acc=-1,-1
            
            loss = self.criterion(prediction,label)
            self.loss = loss
            
            softmax    = self.softmax(prediction).cpu().detach().numpy()
            prediction = torch.argmax(prediction,dim=-1)
            accuracy   = (prediction == label).sum().item() / float(prediction.nelement())        
            prediction = prediction.cpu().detach().numpy()
        
        return {'prediction' : prediction,
                'softmax'    : softmax,
                'loss'       : loss.cpu().detach().item(),
                'accuracy'   : accuracy}

    def backward(self):
        self.opt.zero_grad()  # Reset gradients accumulation
        self.loss.backward()
        self.opt.step()
        


    def save_state(self, prefix='./snapshot'):
        # Output file name
        #filename = '%s-%d.ckpt' % (prefix, self.iteration)
        filename = '%s.ckpt' % (prefix)
    
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.opt.state_dict(),
            'state_dict': self.model.state_dict()
        }, filename)
        return filename

    def restore_state(self,weight_file):
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)
            # load network weights
            self.net.load_state_dict(checkpoint['state_dict'], strict=False)
            # if optim is provided, load the state of the optim
            if self.opt is not None:
                self.opt.load_state_dict(checkpoint['optimizer'])
            # load iteration count
            self.iteration = checkpoint['global_step']
    
        

        
