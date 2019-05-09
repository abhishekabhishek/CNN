"""
START PROGRAM HERE

Script to pass commandline arguments from user to neural net framework.

Author: Julian Ding
"""

# Make sure all custom modules can be seen by the compiler
import os
import sys

par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if par_dir not in sys.path:
    sys.path.append(par_dir)
    
# Let's begin...
import training_utils.engine as net
import models
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', nargs='+', dest='gpu', required=False)
    parser.add_argument('-path', dest='path', required=True)
    parser.add_argument('-vs', dest='val_split', required=True)
    parser.add_argument('-ts', dest='test_split', required=True)
    parser.add_argument('-tnb', dest='batch_size_train', required=True)
    parser.add_argument('-vlb', dest='batch_size_val', required=True)
    parser.add_argument('-tsb', dest='batch_size_test', required=True)
    parser.add_argument('-tsb', dest='batch_size_test', required=True)
    parser.add_argument('-save', dest='save_path', required=False)
    parser.add_argument('-desc', dest='data_description', required=False)
    
    config = parser.parse_args()
    if config.gpu is None:
        config.gpu = False
    if config.save_path is None:
        config.save_path = 'save_path'
    if config.data_description is None:
        config.data_description = 'DATA DESCRIPTION'
        
    return config

if __name__ == '__main__':
    config = main()
    model = models.resnet.resnet18()
    nnet = net.Engine(model, config)
    nnet.train()
    