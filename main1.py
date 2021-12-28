import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import math
import argparse
import resource
import pandas as pd
import numpy as np
import sklearn.metrics
from scipy import stats
from PIL import Image

from PASS import protoAugSSL
from PASS_for_federated import protoAugSSL_for_federated
from ResNet import resnet18_cbam
from myNetwork import network
from iCIFAR100 import iCIFAR100

parser = argparse.ArgumentParser(description='Prototype Augmentation and Self-Supervision for Incremental Learning')
parser.add_argument('--epochs', default=1, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=5, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='cifar100', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=10, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=10, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--protoAug_weight', default=10.0, type=float, help='protoAug loss weight')
parser.add_argument('--kd_weight', default=10.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')


#在联邦学习框架下进行实验的相应设置
parser.add_argument('--client_num', default=2 , type=int, help='the number of clients')
parser.add_argument('--rounds', default=10 , type=int, help='联邦学习轮次')

args = parser.parse_args()
print(args)


def main():
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")


    #为client准备本地模型
    local_model_list = []
    for i in range(args.client_num):
        task_size = 10  # number of classes in each incremental step
        file_name = args.data_name + '_' + str(args.fg_nc) + '_' + "client_index_" +str(i)
        feature_extractor = resnet18_cbam()
        local_model_list.append(protoAugSSL_for_federated(args, file_name, feature_extractor, task_size, device))


    class_set = list(range(20))
    task_size = 10
    #每个client 对自己的模型进行进行预训练
    #在预训练的过程中  会产生初始的原型  全部发往中心保存
    for i in range(args.client_num):

        classes = [i*task_size,(i+1)*task_size]
        local_model_list[i].pre_beforeTrain(classes)
        local_model_list[i].pretrain(2,classes)


    # 中心下发prototype 给每个client
    # 中心下发 别的client的模型给client i
    for i in range(args.client_num):
        local_model_list[i].pre_afterTrain_save()
    for i in range(args.client_num):
        local_model_list[i].pre_afterTrain_load()

    #进行联邦学习的轮次
    for i in range(args.rounds):

        for epoch in range(args.client_num):
            # 中心下发prototype 给每个client
            # 中心下发 别的client的模型给client i 进行知识融合 或者这里我们采用的是 增量学习的方式
            # 拿到上述的数据后 开始进行增量学习
            local_model_list[i].beforeTrain(classes)
            local_model_list[i].train(classes)




        #经过上述的 持续学习，原型 得到了新的更新
        #对从每个client拿到的增量学习后的模型 进行中心聚合,下发 特征提取层，以及得到更新后原型




        return


    ####### Test ######
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    print("############# Test for each Task #############")
    test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    acc_all = []
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.eval()
        acc_up2now = []
        for i in range(current_task+1):
            if i == 0:
                classes = [0, args.fg_nc]
            else:
                classes = [args.fg_nc + (i - 1) * task_size, args.fg_nc + i * task_size]
            test_dataset.getTestData_up2now(classes)
            test_loader = DataLoader(dataset=test_dataset,
                                     shuffle=True,
                                     batch_size=args.batch_size)
            correct, total = 0.0, 0.0
            for setp, (indexs, imgs, labels) in enumerate(test_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(imgs)
                outputs = outputs[:, ::4]
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)
            accuracy = correct.item() / total
            acc_up2now.append(accuracy)
        if current_task < args.task_num:
            acc_up2now.extend((args.task_num-current_task)*[0])
        acc_all.append(acc_up2now)
        print(acc_up2now)
    print(acc_all)

    print("############# Test for up2now Task #############")
    test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.to(device)
        model.eval()

        classes = [0, args.fg_nc + current_task * task_size]
        test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=test_dataset,
                                 shuffle=True,
                                 batch_size=args.batch_size)
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs)
            outputs = outputs[:, ::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        print(accuracy)


if __name__ == "__main__":
    main()