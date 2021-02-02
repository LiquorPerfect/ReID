# -*- encoding: utf-8 -*-
'''
---------------------------------------------- 
* @File    :   test.py
* @Time    :   2021/02/02 22:18:37
* @Author  :   zhang_jinlong
* @Version :   1.0
* @Contact :   zhangjinlongwin@163.com
* @Address ：  https://github.com/LiquorPerfect
-----------------------------------------------
'''
import argparse
import torch
import os
import torchvision


def parser_args():
    parser = argparse.ArgumentParser(
        description="For testing the net's result")
    parser.add_argument("--disable_cuda",
                        action="store_true",
                        help="Disable cuda")
    parser.add_argument("--dataset_dir",
                        default="./",
                        type=str,
                        help="The test images dir path")
    parser.add_argument("--model_dir",
                        default="./",
                        type=str,
                        help="the save model path")
    parser.add_argument("--epoch",
                        default="last",
                        type=str,
                        help="Select the different epoch model")
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="batch_size")
    parser.add_argument("--analysis", action="store_true", help="...")
    parser.add_argument("--pcb",
                        default=True,
                        action="store_true",
                        help="Select the net")
    return parser.parse_args()


def used_device(disable_cuda: bool = False):
    device = torch.device(
        "cude:0" if torch.cuda.is_available and not disable_cuda else "cpu")
    return device


def load_model(model_dir, epoch, device):
    model = torch.load(os.path.join(model_dir, "model_{}.pth".format(epoch)))
    model.to(device)
    model.eval()
    return model


def loag_test_images(model, dataset_path, batch_size):
    data_transforms = getattr(model, "transforms")
    image_datasets = {
        x: torchvision.datasets.ImageFolder(os.path.join(dataset_path, x),
                                            transform=data_transforms[x])
        for x in ["gallery", "query", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=16)
        for x in ["gallery", "quary", "val"]
    }
    return dataloaders
