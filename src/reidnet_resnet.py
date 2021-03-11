# -*- encoding: utf-8 -*-
'''
-----------------------------------------------
* @File    :   reidnet_resnet.py
* @Time    :   2021/03/08 22:05:03
* @Author  :   zhang_jinlong
* @Version :   1.0
* @Contact :   zhangjinlongwin@163.com
* @Address ：  https://github.com/LiquorPerfect
-----------------------------------------------
'''

import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
from torch.optim import lr_scheduler

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ReIDNetResNet(torchvision.models.ResNet):
    def __init__(self, block, layers, num_calsses=751, bottleneck=256):
        super().__init__(block, layers)
        self.num_calsses = num_calsses
        self.avgpool = nn.AdaptiveAvgPool1d((1, 1))

        self.tail = nn.Sequential(
            OrderedDict([("bottleneck", nn.Linear(self.inplanes, bottleneck)),
                         ("bn", nn.BatchNorm1d(self.inplanes, bottleneck)),
                         ("dropout", nn.Dropout(p=0.5))]))
        self.tfc = nn.Linear(bottleneck, self.num_calsses)

        for m in self.tail.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        for m in self.tfc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=0.001)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f = x

        x = self.tail(x)
        x = self.tfc(x)

        return x, f

    def feature_size(self):
        return 2048


def reidnet_resnet_50(model_path=None, num_calsses=751, **kwargs):
    m = ReIDNetResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3],
                      num_calsses, **kwargs)
    if model_path is None:
        m.load_state_dict(model_zoo.load_url(model_urls['resnet50']),
                          strict=False)
    else:
        m.load_state_dict(torch.load(model_path), strict=True)
    return m


def train_reidnet_resnet(device,
                         datasets: dict,
                         model: torch.nn.Module,
                         criterion,
                         lr,
                         batch_size: int,
                         num_epochs=60,
                         path="./"):
    since = time.time()

    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="error")

    ignored_params = list(map(id, model.tfc.parameters())) + list(
        map(id, model.tail.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    optimizer = torch.optim.SGD([{
        "params": base_params,
        'lr': 0.1 * lr
    }, {
        "params": model.tfc.parameters(),
        "lr": lr
    }, {
        "params": model.tail.parameters(),
        "lr": lr
    }],
                                weight_decay=5e-4,
                                momentum=0.9,
                                nesterov=True)

    #Decay LR by a factor of 0.1 every 40 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    if not os.path.isdir(path):
        os.makedirs(path)

    model = model.to(device)

    y_loss = {"train": [], "val": []}
    y_err = {"train": [], "val": []}
    x_epoch = []

    dataloaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                       shuffle=True, num_workers=8)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        x_epoch.append(epoch)

        #each epoch has a training and validation phase

        for phase in ['train', 'val']:
            if phase == "train":
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            #Iterate over data.
            for data in dataloaders[phase]:
                #get the input
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape

                if now_batch_size < batch_size:
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs, features = model(inputs)
                else:
                    outputs, features = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * now_batch_size

                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)

            # deep copy the model
            if phase == 'val':
                ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
                ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
                ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
                ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
                if epoch == 0:
                    ax0.legend()
                    ax1.legend()

                fig.savefig(os.path.join(path, 'train.jpg'))

                if epoch % 10 == 9:
                    save_path = os.path.join(
                        path, 'reidnet_resnet_{}.pth'.format(epoch))
                    torch.save(model.state_dict(), save_path)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    save_path = os.path.join(path, 'model_{}.pth'.format('last'))
    torch.save(model, save_path)

    return model
