# -*- encoding: utf-8 -*-
'''
---------------------------------------------- 
* @File    :   train.py
* @Time    :   2021/01/03 22:29:48
* @Author  :   zhang_jinlong
* @Version :   1.0
* @Contact :   zhangjinlongwin@163.com
* @Address ：  https://github.com/LiquorPerfect
-----------------------------------------------
'''

import argparse
import os
import shutil

import torch
import torch.backends.cudnn as cudnn
import torchvision

import reidnet_pcb
import reidnet_resnet


#define the required hyperparameters
def args():
    parser = argparse.ArgumentParser(
        description="The ReID Model Training Example")
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Disable CUDA')
    parser.add_argument("--model_dir",
                        default="F:/ReID/reid_myself/model",
                        type=str,
                        help="The output model save dir")
    parser.add_argument(
        "--data_dir",
        default="F:/ReID/reid_myself/data/Market-1501-v15.09.15",
        type=str,
        help="The trian and val datasets dir")
    parser.add_argument("--batch_size",
                        default=24,
                        type=int,
                        help='batch_size')
    parser.add_argument("--lr", default=0.04, type=float, help='learning rate')
    parser.add_argument("--pcb",
                        default=True,
                        action="store_true",
                        help="you can select pcb or resnet")
    return parser.parse_args()


#specify gpu or cpu for training
#disable_cuda:bool = False
def used_device(disable_cuda: bool = False):
    if torch.cuda.is_available() and not disable_cuda:
        device = torch.device("cuda:0")
        #use cudnn accelerate the net
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    return device


# preprocess the datasets
def dataset_preprocess(dataset_path):
    transforms = {
        "train":
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(384, 192), interpolation=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ]),
        "val":
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(384, 192), interpolation=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ]),
        "gallery":
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(384, 192), interpolation=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ]),
        "query":
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(384, 192), interpolation=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
    }
    datasets = {
        x: torchvision.datasets.ImageFolder(os.path.join(dataset_path, x),
                                            transform=transforms[x])
        for x in ["train", "val"]
    }
    return datasets


def judge_model_dir(model_dir):
    #judging the model_dir exits
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    #copy py files to model_dir
    src_path = os.path.dirname(os.path.abspath(__file__))
    src_files = os.listdir(src_path)
    for file in src_files:
        if file[-3:] == '.py':
            shutil.copy(os.path.join(src_path, file),
                        os.path.join(model_dir, file))

#create model
def create_model(pcb: bool, datasets):
    if pcb:
        model = reidnet_pcb.PCB(num_classes=len(datasets['train'].classes))
    else:
        model = 's'
    return model


def train():
    opt = args()
    device = used_device(opt.disable_cuda)
    datasets = dataset_preprocess(opt.data_dir)
    judge_model_dir(opt.model_dir)
    model = create_model(opt.pcb, datasets)
    if opt.pcb:
        reidnet_pcb.training_reidnet_pcb(device=device,
                                         datasets=datasets,
                                         model=model,
                                         criterion=torch.nn.CrossEntropyLoss(),
                                         lr=opt.lr,
                                         batch_size=opt.batch_size,
                                         path=opt.model_dir)
    else:
        reidnet_resnet


if __name__ == "__main__":
    train()
