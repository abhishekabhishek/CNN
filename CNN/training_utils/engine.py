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
                    self.model = nn.DataParallel(self.model, device_ids=config.gpu_list, dim=0)

                print("cuda is available")
            else:
                self.device=torch.device("cpu")
                print("cuda is not available")
        else:
            print("will not use gpu")
            self.device=torch.device("cpu")

        print(self.device)

        model.to(self.device)

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
