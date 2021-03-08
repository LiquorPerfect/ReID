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
    return image_datasets, dataloaders


def predict_and_extract_features(data_loader, model, device):
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
        imgflr = torch.index_select(image, 3,
                                    torch.arange(w - 1, 0, -1).to(device))

        with torch.no_grad():
            y1, f1 = model(image)
            y2, f2 = model(imgflr)

            feat = f1 + f2
            sm = torch.nn.Softmax(dim=1)
            if isinstance(model, reidnet_pcb.ReIDNetPCB):
                out = sm(y1[0])
                for i in range(1, len(y1)):
                    out = out + sm(y1[i])

                for i in range(0, len(y2)):
                    out = out + sm(y2[i])

            elif isinstance(model, reidnet_resnet.ReIDNetResNet):
                out = sm(y1) + sm(y2)
            else:
                assert False

            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            if isinstance(model, reidnet_pcb.ReIDNetPCB):
                norm = torch.norm(feat, p=2, dim=1, keepdim=True) * np.sqrt(6)
                feat = feat.div(norm)
                feat = feat.view(feat.size(0), -1)
            elif isinstance(model, reidnet_resnet.ReIDNetResNet):
                norm = torch.norm(feat, p=2, dim=1, keepdim=True)
                feat = feat.div(norm)
            else:
                assert False

            features = torch.cat((features, feat.cpu()), 0)
            _, preds = torch.max(out, 1, keepdim=True)
            classes = torch.cat((classes, preds.cpu()), 0)

    return features, classes


def get_label_and_cam(path):
    labels = []
    cam = []

    for p, v in path:
        p, file_name = os.path.split(p)
        _, label = os.path.split(p)
        labels.append(label)
        cam.append(label + file_name)
    return labels, cam


def dump_test_result(opt, image_datasets, data_loader, model, device):
    gallery_label, gallery_cam = get_label_and_cam(
        image_datasets['gallery'].imgs)
    query_label, quary_cam = get_label_and_cam(image_datasets['query'].imgs)

    gallery_feature, _ = predict_and_extract_features(data_loader['gallery'],
                                                      model, device)
    query_feature, _ = predict_and_extract_features(data_loader['query'],
                                                    model, device)
    # _, prediction = predict_and_extract_features(data_loader['val'], model,
    #  device)
    if opt.analysis:
        data = []
        fig = plt.figure()
        axs = [fig.add_subplot(1, 5, x) for x in range(1, 6)]
        save_id = 0
        for idx, sample in enumerate(image_datasets['val'].imgs):
            path, true_label = sample
            if true_label != prediction[idx]:

                axs[0].imshow(plt.imread(path))
                dir1 = path.split('/')[-2]
                axs[0].set(title=dir1)

                dir2 = image_datasets['val'].classes[prediction[idx]]
                path2 = os.path.join(opt.dataset_dir, 'train', dir2)
                files2 = os.listdir(path2)
                for k in range(0, 4):
                    j = k if k <= len(files2) - 1 else len(files2) - 1

                    img = plt.imread(os.path.join(path2, files2[j]))
                    axs[1 + k].imshow(img)
                    axs[1 + k].set(
                        title=image_datasets['val'].classes[prediction[idx]])

                fig.savefig(
                    os.path.join(opt.model_dir,
                                 'SUS_VAL_{}.jpg'.format(save_id)))
                data.append({
                    "id": "{}".format(save_id),
                    "dir1": dir1,
                    "dir2": dir2,
                    "name1": dir1 + ".jpg",
                    "name2": dir2 + ".jpg"
                })
                save_id = save_id + 1
        plt.close(fig)
        with open(os.path.join(opt.model_dir, 'sus_val.txt'), 'w') as f:
            json.dump(data, f)

    result = {
        'gallery_f': gallery_feature.numpy(),
        'gallery_label': gallery_label,
        'gallery_cam': gallery_cam,
        'gallery_path': image_datasets['gallery'].imgs,
        'query_f': query_feature.numpy(),
        'query_label': query_label,
        'query_cam': query_cam,
        'query_path': image_datasets['query'].imgs
    }

    with open(os.path.join(opt.model_dir, 'test_result.p'), 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


def main():
    args = parser_args()
    device = used_device(args.disable_cuda)
    model = load_model(args.model_dir, args.epoch, device)
    image_datasets, dataloaders = propocess_test_images(
        model, args.dataset_dir, args.batch_size)
    features, classes = predict_and_extract_features(dataloaders, model,
                                                     device)
    labels, cam = get_label_and_cam()
    dump_test_result(args, image_datasets, data_loader, model, device)


if __name__ == "__main__":
    main()