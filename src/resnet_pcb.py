# -*- encoding: utf-8 -*-
'''
---------------------------------------------
@File    :   resnet_pcb.py
@Time    :   2021/01/03 00:13:56
@Author  :   zhang_jinlong
@Version :   1.0
@Contact :   zhangjinlongwin@163.com
@Address ：  https://github.com/LiquorPerfect
---------------------------------------------
'''

import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
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


class ReIDNetPCB(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=751, bottleneck=256):
        super(ReIDNetPCB, self).__init__(block, layers)
        self.num_part = 6
        self.num_calsses = num_classes

        # self.avgpool=nn.AvgPool2d((4,12))
        self.avgpool = nn.AdaptiveAvgPool2d((self.num_part, 1))
        self.dropout = nn.Dropout(p=0.5)

        self.layer4[0].downsample[0].stride = (1, 1)
        self.layer4[0].conv2.stride = (1, 1)

        self.tail1 = self._make_tail(self.inplanes, bottleneck)
        self.tail2 = self._make_tail(self.inplanes, bottleneck)
        self.tail3 = self._make_tail(self.inplanes, bottleneck)
        self.tail4 = self._make_tail(self.inplanes, bottleneck)
        self.tail5 = self._make_tail(self.inplanes, bottleneck)
        self.tail6 = self._make_tail(self.inplanes, bottleneck)

        self.tfc1 = self._make_fc(bottleneck, self.num_calsses)
        self.tfc2 = self._make_fc(bottleneck, self.num_calsses)
        self.tfc3 = self._make_fc(bottleneck, self.num_calsses)
        self.tfc4 = self._make_fc(bottleneck, self.num_calsses)
        self.tfc5 = self._make_fc(bottleneck, self.num_calsses)
        self.tfc6 = self._make_fc(bottleneck, self.num_calsses)

    def _make_tail(self, inplaces, places):
        ret = nn.Sequential(
            OrderedDict([('fc1', nn.Linear(inplaces, places)),
                         ('bn', nn.BatchNorm1d(places)),
                         ('dropout', nn.Dropout(p=0.5))]))
        for mod in ret.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight.data, a=0, mode='fan_out')
                nn.init.constant_(mod.bias.data, 0.0)
            elif isinstance(mod, nn.BatchNorm1d):
                nn.init.normal_(mod.weight.data, 1.0, 0.02)
                nn.init.constant_(mod.bias.data, 0.0)
        return ret

    def _make_fc(self, inplaces, places):
        ret = nn.Sequential(OrderedDict([('fc2', nn.Linear(inplaces,
                                                           places))]))
        for mod in ret.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight.data, std=0.01)
                nn.init.constant_(mod.bias.data, 0.0)
            elif isinstance(mod, nn.BatchNorm1d):
                mod.weight.data.fill(1)
                mod.bias.data.zero_()
        return ret

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

        f = x.detach().squeeze()

        x = self.dropout(x)

        x1 = torch.squeeze(x[:, :, 0])
        x2 = torch.squeeze(x[:, :, 1])
        x3 = torch.squeeze(x[:, :, 2])
        x4 = torch.squeeze(x[:, :, 3])
        x5 = torch.squeeze(x[:, :, 4])
        x6 = torch.squeeze(x[:, :, 5])

        x1 = self.tail1(x1)
        x1 = self.tfc1(x1)
        x2 = self.tail2(x2)
        x2 = self.tfc2(x2)
        x3 = self.tail3(x3)
        x3 = self.tfc3(x3)
        x4 = self.tail4(x4)
        x4 = self.tfc4(x4)
        x5 = self.tail5(x5)
        x5 = self.tfc5(x5)
        x6 = self.tail6(x6)
        x6 = self.tfc6(x6)

        y = [x1, x2, x3, x4, x5, x6]
        return y, f

    def feature_size(self):
        return 2048


def PCB(model_path=None, num_classes=751):
    m = ReIDNetPCB(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3],
                   num_classes)
    if model_path is None:
        m.load_state_dict(model_zoo.load_url(model_urls["resnet50"]),
                          strict=False)
    else:
        m.load_state_dict(torch.load(model_urls), strict=True)
    return m


def training_reidnet_pcb(device,
                         datasets: dict,
                         model: torch.nn.Module,
                         criterion,
                         lr,
                         batch_size: int,
                         num_epochs=60,
                         path='./'):
    since = time.time()
    fig = plt.figure(num="ReID Training")
    ax0 = fig.add_subplot(121, title='loss')
    ax1 = fig.add_subplot(122, title='error')

    ignored_params = []
    ignored_params = ignored_params + list(map(
        id, model.tfc1.parameters())) + list(map(id, model.tail1.parameters()))
    ignored_params = ignored_params + list(map(
        id, model.tfc2.parameters())) + list(map(id, model.tail2.parameters()))
    ignored_params = ignored_params + list(map(
        id, model.tfc3.parameters())) + list(map(id, model.tail3.parameters()))
    ignored_params = ignored_params + list(map(
        id, model.tfc4.parameters())) + list(map(id, model.tail4.parameters()))
    ignored_params = ignored_params + list(map(
        id, model.tfc5.parameters())) + list(map(id, model.tail5.parameters()))
    ignored_params = ignored_params + list(map(
        id, model.tfc6.parameters())) + list(map(id, model.tail6.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())

    optimizer = torch.optim.SGD([
        {
            'params': base_params,
            'lr': 0.1 * lr
        },
        {
            'params': model.tfc1.parameters(),
            'lr': lr
        },
        {
            'params': model.tfc2.parameters(),
            'lr': lr
        },
        {
            'params': model.tfc3.parameters(),
            'lr': lr
        },
        {
            'params': model.tfc4.parameters(),
            'lr': lr
        },
        {
            'params': model.tfc5.parameters(),
            'lr': lr
        },
        {
            'params': model.tfc6.parameters(),
            'lr': lr
        },
        {
            'params': model.tail1.parameters(),
            'lr': lr
        },
        {
            'params': model.tail2.parameters(),
            'lr': lr
        },
        {
            'params': model.tail3.parameters(),
            'lr': lr
        },
        {
            'params': model.tail4.parameters(),
            'lr': lr
        },
        {
            'params': model.tail5.parameters(),
            'lr': lr
        },
        {
            'params': model.tail6.parameters(),
            'lr': lr
        },
    ],
                                weight_decay=5e-4,
                                momentum=0.9,
                                nesterov=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    model = model.to(device)

    y_loss = {'train': [], 'val': []}
    y_err = {'train': [], 'val': []}
    x_epoch = []

    dataloaders = {
        x: torch.utils.data.DataLoader(datasets[x],
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=8)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        x_epoch.append(epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            #Iterate over data
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape
                #skip the last batch
                if now_batch_size < batch_size:
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)

                #zero the parameter gradients
                optimizer.zero_grad()

                #forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs, feature = model(inputs)
                else:
                    outputs, featrue = model(inputs)
                sm = nn.Softmax(dim=1)

                sorce = sm(outputs[0].detach())
                for i in range(1, len(outputs)):
                    sorce = sorce + sm(outputs[i].detach())

                _, preds = torch.max(sorce, 1)
                loss = criterion(outputs[0], labels)
                for i in range(1, len(outputs)):
                    loss = loss + criterion(outputs[i], labels)

                #backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(preds == labels))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss,
                                                       epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1 - epoch_acc)
            #deep copy the model
            if phase == "val":
                ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
                ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
                ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
                ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
                if epoch == 0:
                    ax0.legend()
                    ax1.legend()

                fig.savefig(os.path.join(path, 'train.jpg'))

                if epoch % 10 == 0:
                    save_path = os.path.join(
                        path, 'reidnet_pcb_{}.pth'.format(epoch))
                    torch.save(model.state_dict(), save_path)
        
        time_elapsed = time.time() - since
        print("Time compelet in {:.0f}m {:.0f}".format(time_elapsed // 60,
                                                   time_elapsed % 60))
    time_elapsed = time.time() - since
    print("Time compelet in {:.0f}m {:.0f}".format(time_elapsed // 60,
                                                   time_elapsed % 60))

    #load the last model net and weights
    save_path = os.path.join(path, 'model_{}.pth'.format('last'))
    torch.save(model, save_path)
    return model
