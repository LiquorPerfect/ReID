# -*- encoding: utf-8 -*-
'''
-----------------------------------------------
* @File    :   test.py
* @Time    :   2021/02/02 22:18:37
* @Author  :   zhang_jinlong
* @Version :   1.0
* @Contact :   zhangjinlongwin@163.com
* @Address ：  https://github.com/LiquorPerfect
-----------------------------------------------
'''

import argparse
import os

import torch
import torchvision

import reidnet_pcb
import reidnet_resnet


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


def propocess_test_images(model, dataset_path, batch_size):
    #这边需要打印一下 model中有什么东西
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


def predict_and_extract_features(data_loader, device):
    features = torch.FloatTensor()  #这边这个 表示什么意思
    classes = torch.LongTensor()  #如果没有给出参数，则返回空的零维张量
    #这边相当于初始化

    for data in data_loader:
        image, lable = data
        n, c, h, w = image.size()

        image = image.to(device)
        #第一个参数是索引的对象，第二个参数0表示按行索引，
        #1表示按列进行索引，第三个参数是一个tensor，就是索引的序
        #表示倒序，将图片反转了
        imgflr = torch.index_select(image, 3, torch.arange(w - 1, 0,
                                                           -1)).to(device)

        with torch.no_grad():
            y1, f1 = model(image)
            y2, f2 = model(imgflr)

            feat = f1 + f2
            sm = torch.nn.Softmax(dim=1)
            if isinstance(model, reidnet_pcb.ReIDNetPCB):
                norm = torch.norm(feat, p=2, dim=1, keepidm=True) * np.sqrt(6)
                feat = feat.div(norm)
                feat = feat.view(feat.size(0), -1)
            elif isinstance(model, reidnet_resnet.ReIDNetResNet):
                norm = torch.norm(feat, p=2, dim=1, keepdim=True)
                feat = feat.div(norm)
            else:
                assert False

            featrues = torch.cat((featrues, feat.cpu()), 0)
            _, preds = torch.max(out, 1, keepdim=True)
            classes = torch.cat((classes, preds.cpu()), 0)

    return features, classes