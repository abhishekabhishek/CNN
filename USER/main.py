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
import models.resnet as resnet
import argparse

def main():
    parser = argparse.ArgumentParser(
            description= '[HK-Canada] TRIUMF Neutrino Group: Welcome to the deep learning interface! Collaborators: Wojciech Fedorko, Julian Ding, Abhishek Kajal',
            epilog= 'Happy training!')
    parser.add_argument('-device', dest='device', default='cpu',
                        required=False, help='Enter cpu to use CPU resources or gpu to use GPU resources.')
    parser.add_argument('-gpu', nargs='+', dest='gpu_list', default=None,
                        required=False, help='List of available GPUs')
    parser.add_argument('-path', dest='path',
                        required=True, help='Path to training dataset.')
    parser.add_argument('-vs', dest='val_split',
                        required=True, help='Fraction of dataset used in validation. Note: requires vs + ts < 1')
    parser.add_argument('-ts', dest='test_split',
                        required=True, help='Fraction of dataset used in testing. Note: requires vs + ts < 1')
    parser.add_argument('-tnb', dest='batch_size_train',
                        required=True, help='Batch size for training.')
    parser.add_argument('-vlb', dest='batch_size_val',
                        required=True, help='Batch size for validating.')
    parser.add_argument('-tsb', dest='batch_size_test',
                        required=True, help='Batch size for testing.')
    parser.add_argument('-save', dest='save_path', default='save_path',
                        required=False, help='Specify path to save data to. Default is save_path')
    parser.add_argument('-desc', dest='data_description', default='DATA DESCRIPTION',
                        required=False, help='Specify description for data.')
    
    config = parser.parse_args()
    if config.gpu_list is None:
        config.gpu = False
        
    return config

if __name__ == '__main__':
    config = main()
    model = resnet.resnet18()
    nnet = net.Engine(model, config)
    nnet.train()
    