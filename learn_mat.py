import io
import scipy.io as matio
import os.path
import numpy as np
import time
import torch


root_path = '../VireoFood172/SplitAndIngreLabel/'


# #load input matrix (data,ingredient)
# train_label=  matio.loadmat(root_path+ 'train_label.mat')
# print(train_label)
# wordIndicator_test = matio.loadmat(root_path+ 'ingredient_all_feature.mat')['ingredient_all_feature']
# print(wordIndicator_test.size)
# wordIndicator_test = matio.loadmat(root_path+ 'ingredient_train_feature.mat')['ingredient_train_feature']
# print(wordIndicator_test.size)
# # train_label = matio.loadmat(root_path+'train_label.mat')
# # print(train_label)

def build_labels(root_path,train_data_path,mode):
    with io.open(train_data_path, encoding='utf-8') as file:
        lines = file.read().split('\n')[:-1]

    num_img = len(lines)
    #train_label = np.zeros(num_img)

    i = 0
    train_label = []
    for line in lines:
        label = line.split('/')
        print(label)
        label = int(label[1])
        train_label.append(label)
    
    train_label = np.array(train_label,dtype = 'int64')
    if mode == 'val':
        matio.savemat(root_path + mode +'_label.mat', {'validation_label': train_label})
    else :
        matio.savemat(root_path + mode +'_label.mat', {mode + '_label': train_label})
    return train_label

def create_labels(root_path):
    train_data_path = root_path + 'TR.txt'
    val_data_path = root_path + 'VAL.txt'
    test_data_path = root_path + 'TE.txt'

    train_label = build_labels(root_path,train_data_path,'train')
    val_label = build_labels(root_path,train_data_path,'val')
    test_label = build_labels(root_path,train_data_path,'test')

    return train_label,val_label,test_label


# create_labels(root_path)

# train_label=  matio.loadmat(root_path+ 'train_label.mat')
# print(train_label)